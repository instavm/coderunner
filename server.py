# --- IMPORTS ---
import asyncio
import base64
import binascii
import json
import logging
import os
import pathlib
import time
import uuid
from typing import Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
from datetime import datetime, timedelta

import aiofiles
import websockets
import httpx
# Import Context for progress reporting
from mcp.server.fastmcp import FastMCP, Context
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import socket
# --- CONFIGURATION & SETUP ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize the MCP server with a descriptive name for the toolset
mcp = FastMCP("CodeRunner")

# Kernel pool configuration
MAX_KERNELS = 5
MIN_KERNELS = 2
KERNEL_TIMEOUT = 300  # 5 minutes
KERNEL_HEALTH_CHECK_INTERVAL = 30  # 30 seconds
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 2  # exponential backoff base

# Jupyter connection settings
JUPYTER_WS_URL = "ws://127.0.0.1:8888"
JUPYTER_HTTP_URL = "http://127.0.0.1:8888"

# Enhanced WebSocket settings
WEBSOCKET_TIMEOUT = 600  # 10 minutes for long operations
WEBSOCKET_PING_INTERVAL = 30
WEBSOCKET_PING_TIMEOUT = 10

# Directory configuration (ensure this matches your Jupyter/Docker setup)
# This directory must be accessible by both this script and the Jupyter kernel.
SHARED_DIR = pathlib.Path("/app/uploads")
SHARED_DIR.mkdir(exist_ok=True)
KERNEL_ID_FILE_PATH = SHARED_DIR / "python_kernel_id.txt"

def resolve_with_system_dns(hostname):
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror as e:
        print(f"Error resolving {hostname}: {e}")
        return None

PLAYWRIGHT_WS_URL =f"ws://127.0.0.1:3000/"

# --- KERNEL MANAGEMENT CLASSES ---

class KernelState(Enum):
    HEALTHY = "healthy"
    BUSY = "busy"
    UNRESPONSIVE = "unresponsive"
    FAILED = "failed"
    RESTARTING = "restarting"

@dataclass
class KernelInfo:
    kernel_id: str
    state: KernelState = KernelState.HEALTHY
    last_used: datetime = field(default_factory=datetime.now)
    last_health_check: datetime = field(default_factory=datetime.now)
    current_operation: Optional[str] = None
    failure_count: int = 0
    websocket: Optional[object] = None
    
    def is_available(self) -> bool:
        return self.state == KernelState.HEALTHY
    
    def is_stale(self) -> bool:
        return datetime.now() - self.last_used > timedelta(seconds=KERNEL_TIMEOUT)
    
    def needs_health_check(self) -> bool:
        return datetime.now() - self.last_health_check > timedelta(seconds=KERNEL_HEALTH_CHECK_INTERVAL)

class KernelPool:
    def __init__(self):
        self.kernels: Dict[str, KernelInfo] = {}
        self.lock = asyncio.Lock()
        self.busy_kernels: Set[str] = set()
        self._initialized = False
        self._health_check_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the kernel pool with minimum number of kernels"""
        if self._initialized:
            return
            
        async with self.lock:
            logger.info("Initializing kernel pool...")
            
            # Try to use existing kernel first
            existing_kernel = await self._get_existing_kernel()
            if existing_kernel:
                self.kernels[existing_kernel] = KernelInfo(kernel_id=existing_kernel)
                logger.info(f"Added existing kernel to pool: {existing_kernel}")
            
            # Create additional kernels to reach minimum
            while len(self.kernels) < MIN_KERNELS:
                kernel_id = await self._create_new_kernel()
                if kernel_id:
                    self.kernels[kernel_id] = KernelInfo(kernel_id=kernel_id)
                    logger.info(f"Created new kernel: {kernel_id}")
                else:
                    logger.warning("Failed to create minimum number of kernels")
                    break
            
            self._initialized = True
            # Start health check background task
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info(f"Kernel pool initialized with {len(self.kernels)} kernels")
    
    async def get_available_kernel(self) -> Optional[str]:
        """Get an available kernel from the pool"""
        if not self._initialized:
            await self.initialize()
        
        async with self.lock:
            # Find healthy, available kernel
            for kernel_id, kernel_info in self.kernels.items():
                if kernel_info.is_available() and kernel_id not in self.busy_kernels:
                    self.busy_kernels.add(kernel_id)
                    kernel_info.state = KernelState.BUSY
                    kernel_info.last_used = datetime.now()
                    logger.info(f"Assigned kernel {kernel_id} to operation")
                    return kernel_id
            
            # No available kernels, try to create a new one if under limit
            if len(self.kernels) < MAX_KERNELS:
                kernel_id = await self._create_new_kernel()
                if kernel_id:
                    kernel_info = KernelInfo(kernel_id=kernel_id, state=KernelState.BUSY)
                    self.kernels[kernel_id] = kernel_info
                    self.busy_kernels.add(kernel_id)
                    logger.info(f"Created and assigned new kernel: {kernel_id}")
                    return kernel_id
            
            logger.warning("No available kernels in pool")
            return None
    
    async def release_kernel(self, kernel_id: str, failed: bool = False):
        """Release a kernel back to the pool"""
        async with self.lock:
            if kernel_id in self.busy_kernels:
                self.busy_kernels.remove(kernel_id)
            
            if kernel_id in self.kernels:
                kernel_info = self.kernels[kernel_id]
                if failed:
                    kernel_info.failure_count += 1
                    kernel_info.state = KernelState.FAILED
                    logger.warning(f"Kernel {kernel_id} marked as failed (failures: {kernel_info.failure_count})")
                    
                    # Remove failed kernel if it has too many failures
                    if kernel_info.failure_count >= MAX_RETRY_ATTEMPTS:
                        await self._remove_kernel(kernel_id)
                        # Create replacement kernel
                        new_kernel_id = await self._create_new_kernel()
                        if new_kernel_id:
                            self.kernels[new_kernel_id] = KernelInfo(kernel_id=new_kernel_id)
                else:
                    kernel_info.state = KernelState.HEALTHY
                    kernel_info.current_operation = None
                    logger.info(f"Released kernel {kernel_id} back to pool")
    
    async def _get_existing_kernel(self) -> Optional[str]:
        """Try to get kernel ID from existing file"""
        try:
            if os.path.exists(KERNEL_ID_FILE_PATH):
                with open(KERNEL_ID_FILE_PATH, 'r') as file:
                    kernel_id = file.read().strip()
                    if kernel_id and await self._check_kernel_health(kernel_id):
                        return kernel_id
        except Exception as e:
            logger.warning(f"Could not read existing kernel: {e}")
        return None
    
    async def _create_new_kernel(self) -> Optional[str]:
        """Create a new Jupyter kernel"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{JUPYTER_HTTP_URL}/api/kernels",
                    json={"name": "python3"},
                    timeout=30.0
                )
                if response.status_code == 201:
                    kernel_data = response.json()
                    kernel_id = kernel_data["id"]
                    logger.info(f"Created new kernel: {kernel_id}")
                    return kernel_id
                else:
                    logger.error(f"Failed to create kernel: {response.status_code}")
        except Exception as e:
            logger.error(f"Error creating kernel: {e}")
        return None
    
    async def _remove_kernel(self, kernel_id: str):
        """Remove and shutdown a kernel"""
        try:
            async with httpx.AsyncClient() as client:
                await client.delete(
                    f"{JUPYTER_HTTP_URL}/api/kernels/{kernel_id}",
                    timeout=10.0
                )
            logger.info(f"Removed kernel: {kernel_id}")
        except Exception as e:
            logger.warning(f"Error removing kernel {kernel_id}: {e}")
        
        if kernel_id in self.kernels:
            del self.kernels[kernel_id]
        if kernel_id in self.busy_kernels:
            self.busy_kernels.remove(kernel_id)
    
    async def _check_kernel_health(self, kernel_id: str) -> bool:
        """Check if a kernel is healthy by sending a simple command"""
        try:
            jupyter_ws_url = f"{JUPYTER_WS_URL}/api/kernels/{kernel_id}/channels"
            async with websockets.connect(
                jupyter_ws_url,
                ping_interval=WEBSOCKET_PING_INTERVAL,
                ping_timeout=WEBSOCKET_PING_TIMEOUT
            ) as ws:
                # Send simple health check command
                msg_id, request_json = create_jupyter_request("1+1")
                await ws.send(request_json)
                
                # Wait for response with timeout
                start_time = time.time()
                while time.time() - start_time < 10:  # 10 second timeout for health check
                    try:
                        message_str = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        message_data = json.loads(message_str)
                        parent_msg_id = message_data.get("parent_header", {}).get("msg_id")
                        
                        if parent_msg_id == msg_id:
                            msg_type = message_data.get("header", {}).get("msg_type")
                            if msg_type == "status" and message_data.get("content", {}).get("execution_state") == "idle":
                                return True
                    except asyncio.TimeoutError:
                        continue
            return False
        except Exception as e:
            logger.warning(f"Health check failed for kernel {kernel_id}: {e}")
            return False
    
    async def _health_check_loop(self):
        """Background task to monitor kernel health"""
        while True:
            try:
                await asyncio.sleep(KERNEL_HEALTH_CHECK_INTERVAL)
                async with self.lock:
                    unhealthy_kernels = []
                    for kernel_id, kernel_info in self.kernels.items():
                        if kernel_info.needs_health_check() and kernel_id not in self.busy_kernels:
                            if await self._check_kernel_health(kernel_id):
                                kernel_info.last_health_check = datetime.now()
                                kernel_info.state = KernelState.HEALTHY
                            else:
                                kernel_info.state = KernelState.UNRESPONSIVE
                                unhealthy_kernels.append(kernel_id)
                    
                    # Remove unhealthy kernels and create replacements
                    for kernel_id in unhealthy_kernels:
                        logger.warning(f"Removing unhealthy kernel: {kernel_id}")
                        await self._remove_kernel(kernel_id)
                        # Create replacement if below minimum
                        if len(self.kernels) < MIN_KERNELS:
                            new_kernel_id = await self._create_new_kernel()
                            if new_kernel_id:
                                self.kernels[new_kernel_id] = KernelInfo(kernel_id=new_kernel_id)
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

# Global kernel pool instance
kernel_pool = KernelPool()



# --- HELPER FUNCTION ---
def create_jupyter_request(code: str) -> tuple[str, str]:
    """
    Creates a Jupyter execute_request message.
    Returns a tuple: (msg_id, json_payload_string)
    """
    msg_id = uuid.uuid4().hex
    session_id = uuid.uuid4().hex

    request = {
        "header": {
            "msg_id": msg_id,
            "username": "mcp_client",
            "session": session_id,
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        "buffers": [],
    }
    return msg_id, json.dumps(request)


# --- ENHANCED EXECUTION WITH RETRY LOGIC ---

async def execute_with_retry(command: str, ctx: Context, max_attempts: int = MAX_RETRY_ATTEMPTS) -> str:
    """Execute code with retry logic and exponential backoff"""
    last_error = None
    
    for attempt in range(max_attempts):
        try:
            # Get kernel from pool
            kernel_id = await kernel_pool.get_available_kernel()
            if not kernel_id:
                raise Exception("No available kernels in pool")
            
            try:
                result = await _execute_on_kernel(kernel_id, command, ctx)
                # Release kernel back to pool on success
                await kernel_pool.release_kernel(kernel_id, failed=False)
                return result
            except Exception as e:
                # Release kernel as failed
                await kernel_pool.release_kernel(kernel_id, failed=True)
                raise e
                
        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                backoff_time = RETRY_BACKOFF_BASE ** attempt
                logger.warning(f"Execution attempt {attempt + 1} failed: {e}. Retrying in {backoff_time}s...")
                await asyncio.sleep(backoff_time)
            else:
                logger.error(f"All {max_attempts} execution attempts failed. Last error: {e}")
    
    return f"Error: Execution failed after {max_attempts} attempts. Last error: {str(last_error)}"

async def _execute_on_kernel(kernel_id: str, command: str, ctx: Context) -> str:
    """Execute code on a specific kernel with enhanced timeout handling"""
    jupyter_ws_url = f"{JUPYTER_WS_URL}/api/kernels/{kernel_id}/channels"
    final_output_lines = []
    sent_msg_id = None

    try:
        # Enhanced WebSocket connection with longer timeouts
        async with websockets.connect(
            jupyter_ws_url,
            ping_interval=WEBSOCKET_PING_INTERVAL,
            ping_timeout=WEBSOCKET_PING_TIMEOUT,
            close_timeout=10
        ) as jupyter_ws:
            sent_msg_id, jupyter_request_json = create_jupyter_request(command)
            await jupyter_ws.send(jupyter_request_json)
            logger.info(f"Sent execute_request to kernel {kernel_id} (msg_id: {sent_msg_id})")

            execution_complete = False
            start_time = time.time()
            last_activity = start_time
            
            # Progress reporting for long operations
            await ctx.report_progress(progress=f"Executing on kernel {kernel_id[:8]}...")

            while not execution_complete and (time.time() - start_time) < WEBSOCKET_TIMEOUT:
                try:
                    # Adaptive timeout based on recent activity
                    current_time = time.time()
                    time_since_activity = current_time - last_activity
                    
                    # Use shorter timeout if no recent activity, longer if active
                    recv_timeout = 30.0 if time_since_activity > 60 else 5.0
                    
                    message_str = await asyncio.wait_for(jupyter_ws.recv(), timeout=recv_timeout)
                    last_activity = current_time
                    
                except asyncio.TimeoutError:
                    # Send periodic progress updates during long operations
                    elapsed = time.time() - start_time
                    await ctx.report_progress(progress=f"Still executing... ({elapsed:.0f}s elapsed)")
                    continue

                try:
                    message_data = json.loads(message_str)
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON from kernel {kernel_id}")
                    continue
                    
                parent_msg_id = message_data.get("parent_header", {}).get("msg_id")

                if parent_msg_id != sent_msg_id:
                    continue

                msg_type = message_data.get("header", {}).get("msg_type")
                content = message_data.get("content", {})

                if msg_type == "stream":
                    stream_text = content.get("text", "")
                    final_output_lines.append(stream_text)
                    # Stream output as progress
                    await ctx.report_progress(progress=stream_text.strip())

                elif msg_type in ["execute_result", "display_data"]:
                    result_text = content.get("data", {}).get("text/plain", "")
                    final_output_lines.append(result_text)
                    
                elif msg_type == "error":
                    error_traceback = "\n".join(content.get("traceback", []))
                    logger.error(f"Execution error on kernel {kernel_id} for msg_id {sent_msg_id}:\n{error_traceback}")
                    raise Exception(f"Execution Error:\n{error_traceback}")

                elif msg_type == "status" and content.get("execution_state") == "idle":
                    execution_complete = True
                    await ctx.report_progress(progress="Execution completed")

            if not execution_complete:
                elapsed = time.time() - start_time
                timeout_msg = f"Execution timed out after {elapsed:.0f} seconds on kernel {kernel_id}"
                logger.error(f"Execution timed out for msg_id: {sent_msg_id}")
                raise Exception(timeout_msg)

            return "".join(final_output_lines) if final_output_lines else "[Execution successful with no output]"

    except websockets.exceptions.ConnectionClosed as e:
        error_msg = f"WebSocket connection to kernel {kernel_id} closed unexpectedly: {e}"
        logger.error(error_msg)
        raise Exception(error_msg)
    except websockets.exceptions.WebSocketException as e:
        error_msg = f"WebSocket error with kernel {kernel_id}: {e}"
        logger.error(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        logger.error(f"Unexpected error during execution on kernel {kernel_id}: {e}", exc_info=True)
        raise e

# --- MCP TOOLS ---
@mcp.tool()
async def execute_python_code(command: str, ctx: Context) -> str:
    """
    Executes a string of Python code in a persistent Jupyter kernel and returns the final output.
    Uses kernel pool management with automatic retry and recovery for long-running operations.
    Streams intermediate output (stdout) as progress updates.

    Args:
        command: The Python code to execute as a single string.
        ctx: The MCP Context object, used for reporting progress.
    """
    try:
        # Initialize kernel pool if not already done
        if not kernel_pool._initialized:
            await ctx.report_progress(progress="Initializing kernel pool...")
            await kernel_pool.initialize()
        
        # Execute with retry logic
        result = await execute_with_retry(command, ctx)
        return result
        
    except Exception as e:
        logger.error(f"Fatal error in execute_python_code: {e}", exc_info=True)
        return f"Error: Failed to execute code: {str(e)}"

@mcp.tool()
async def navigate_and_get_all_visible_text(url: str) -> str:
    """
    Retrieves all visible text from the entire webpage using Playwright.

    Args:
        url: The URL of the webpage from which to retrieve text.
    """
    # This function doesn't have intermediate steps, so it only needs 'return'.
    try:
        # Note: 'async with async_playwright() as p:' can be slow.
        # For performance, consider managing a single Playwright instance
        # outside the tool function if this tool is called frequently.
        async with async_playwright() as p:
            browser = await p.chromium.connect(PLAYWRIGHT_WS_URL)
            page = await browser.new_page()
            await page.goto(url)

            html_content = await page.content()
            soup = BeautifulSoup(html_content, 'html.parser')
            visible_text = soup.get_text(separator="\n", strip=True)

            await browser.close()

            # The operation is complete, return the final result.
            return visible_text

    except Exception as e:
        logger.error(f"Failed to retrieve all visible text: {e}")
        # An error occurred, return the final error message.
        return f"Error: Failed to retrieve all visible text: {str(e)}"


# Use the streamable_http_app as it's the modern standard
app = mcp.streamable_http_app()