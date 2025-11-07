# --- IMPORTS ---
import asyncio
import base64
import binascii
import json
import logging
import os
import zipfile
import pathlib
import time
import uuid
from typing import Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

import aiofiles
import websockets
import httpx
# Import Context for progress reporting
from mcp.server.fastmcp import FastMCP, Context
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import socket
from starlette.requests import Request
from starlette.responses import JSONResponse
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

# Skills directory configuration
SKILLS_DIR = SHARED_DIR / "skills"
PUBLIC_SKILLS_DIR = SKILLS_DIR / "public"
USER_SKILLS_DIR = SKILLS_DIR / "user"

def resolve_with_system_dns(hostname):
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror as e:
        print(f"Error resolving {hostname}: {e}")
        return None

PLAYWRIGHT_WS_URL =f"ws://127.0.0.1:3000/"

# --- CUSTOM EXCEPTIONS ---

class KernelError(Exception):
    """Base exception for kernel-related errors"""
    pass

class NoKernelAvailableError(KernelError):
    """Raised when no kernels are available in the pool"""
    pass

class KernelExecutionError(KernelError):
    """Raised when kernel execution fails"""
    pass

class KernelTimeoutError(KernelError):
    """Raised when kernel operation times out"""
    pass

# --- KERNEL MANAGEMENT CLASSES ---

class KernelState(Enum):
    HEALTHY = "healthy"
    BUSY = "busy"
    UNRESPONSIVE = "unresponsive"
    FAILED = "failed"

@dataclass
class KernelInfo:
    kernel_id: str
    state: KernelState = KernelState.HEALTHY
    last_used: datetime = field(default_factory=datetime.now)
    last_health_check: datetime = field(default_factory=datetime.now)
    current_operation: Optional[str] = None
    failure_count: int = 0

    def is_available(self) -> bool:
        return self.state == KernelState.HEALTHY

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
            async with aiofiles.open(KERNEL_ID_FILE_PATH, mode='r') as f:
                kernel_id = (await f.read()).strip()
                if kernel_id and await self._check_kernel_health(kernel_id):
                    return kernel_id
        except FileNotFoundError:
            # This is a normal case if the server is starting for the first time.
            pass
        except Exception as e:
            logger.warning(f"Could not read or validate existing kernel from {KERNEL_ID_FILE_PATH}: {e}")
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
                raise NoKernelAvailableError("No available kernels in pool")

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
            await ctx.report_progress(progress=10, message=f"Executing on kernel {kernel_id[:8]}...")

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
                    await ctx.report_progress(progress=30, message=f"Still executing... ({elapsed:.0f}s elapsed)")
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
                    await ctx.report_progress(progress=50, message=stream_text.strip())

                elif msg_type in ["execute_result", "display_data"]:
                    result_text = content.get("data", {}).get("text/plain", "")
                    final_output_lines.append(result_text)

                elif msg_type == "error":
                    error_traceback = "\n".join(content.get("traceback", []))
                    logger.error(f"Execution error on kernel {kernel_id} for msg_id {sent_msg_id}:\n{error_traceback}")
                    raise KernelExecutionError(f"Execution Error:\n{error_traceback}")

                elif msg_type == "status" and content.get("execution_state") == "idle":
                    execution_complete = True
                    await ctx.report_progress(progress=100, message="Execution completed")

            if not execution_complete:
                elapsed = time.time() - start_time
                timeout_msg = f"Execution timed out after {elapsed:.0f} seconds on kernel {kernel_id}"
                logger.error(f"Execution timed out for msg_id: {sent_msg_id}")
                raise KernelTimeoutError(timeout_msg)

            return "".join(final_output_lines) if final_output_lines else "[Execution successful with no output]"

    except websockets.exceptions.ConnectionClosed as e:
        error_msg = f"WebSocket connection to kernel {kernel_id} closed unexpectedly: {e}"
        logger.error(error_msg)
        raise KernelError(error_msg)
    except websockets.exceptions.WebSocketException as e:
        error_msg = f"WebSocket error with kernel {kernel_id}: {e}"
        logger.error(error_msg)
        raise KernelError(error_msg)
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
            await ctx.report_progress(progress=10, message="Initializing kernel pool...")
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


# --- SKILLS MANAGEMENT TOOLS ---


async def _parse_skill_frontmatter(skill_md_path):
    try:
        async with aiofiles.open(skill_md_path, mode='r') as f:
            content = await f.read()
            frontmatter = []
            in_frontmatter = False
            for line in content.splitlines():
                if line.strip() == '---':
                    if in_frontmatter:
                        break
                    else:
                        in_frontmatter = True
                        continue
                if in_frontmatter:
                    frontmatter.append(line)
            
            metadata = {}
            for line in frontmatter:
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
            return metadata
    except Exception:
        return {}

@mcp.tool()
async def list_skills() -> str:
    """
    Lists all available skills in the CodeRunner container.

    Returns a list of available skills organized by category (public/user).
    Public skills are built into the container, while user skills are added by users.

    Returns:
        JSON string with skill names organized by category.
    """
    try:
        # Unzip any user-provided skills
        if USER_SKILLS_DIR.exists():
            for item in USER_SKILLS_DIR.iterdir():
                if item.is_file() and item.suffix == '.zip':
                    with zipfile.ZipFile(item, 'r') as zip_ref:
                        zip_ref.extractall(USER_SKILLS_DIR)
                    os.remove(item)

        skills = {
            "public": [],
            "user": []
        }

        # Helper to process a skills directory
        async def process_skill_dir(directory, category):
            if directory.exists():
                for skill_dir in directory.iterdir():
                    if skill_dir.is_dir():
                        skill_md_path = skill_dir / "SKILL.md"
                        if skill_md_path.exists():
                            metadata = await _parse_skill_frontmatter(skill_md_path)
                            skills[category].append({
                                "name": metadata.get("name", skill_dir.name),
                                "description": metadata.get("description", "No description available.")
                            })

        await process_skill_dir(PUBLIC_SKILLS_DIR, "public")
        await process_skill_dir(USER_SKILLS_DIR, "user")

        # Sort for consistent output
        skills["public"].sort(key=lambda x: x['name'])
        skills["user"].sort(key=lambda x: x['name'])

        result = f"Available Skills:\n\n"
        result += f"Public Skills ({len(skills['public'])}):\n"
        if skills["public"]:
            for skill in skills["public"]:
                result += f"  - {skill['name']}: {skill['description']}\n"
        else:
            result += "  (none)\n"

        result += f"\nUser Skills ({len(skills['user'])}):\n"
        if skills["user"]:
            for skill in skills["user"]:
                result += f"  - {skill['name']}: {skill['description']}\n"
        else:
            result += "  (none)\n"

        result += f"\nUse get_skill_info(skill_name) to read documentation for a specific skill."

        return result

    except Exception as e:
        logger.error(f"Failed to list skills: {e}")
        return f"Error: Failed to list skills: {str(e)}"


async def _read_skill_file(skill_name: str, filename: str) -> tuple[str, str, str]:
    """
    Helper function to read a file from a skill's directory.

    Args:
        skill_name: The name of the skill
        filename: The name of the file to read (e.g., 'SKILL.md', 'EXAMPLES.md')

    Returns:
        A tuple of (content, skill_type, error_message)
        If successful, error_message is None
        If failed, content and skill_type are None
    """
    try:
        # Check public skills first
        public_skill_file = PUBLIC_SKILLS_DIR / skill_name / filename
        user_skill_file = USER_SKILLS_DIR / skill_name / filename

        skill_file_path = None
        skill_type = None

        if public_skill_file.exists():
            skill_file_path = public_skill_file
            skill_type = "public"
        elif user_skill_file.exists():
            skill_file_path = user_skill_file
            skill_type = "user"
        else:
            return None, None, f"Error: File '{filename}' not found in skill '{skill_name}'. Use list_skills() to see available skills."

        # Read the file content
        async with aiofiles.open(skill_file_path, mode='r') as f:
            content = await f.read()

        # Replace all occurrences of /mnt/user-data with /app/uploads
        content = content.replace('/mnt/user-data', '/app/uploads')

        return content, skill_type, None

    except Exception as e:
        logger.error(f"Failed to read file '{filename}' from skill '{skill_name}': {e}")
        return None, None, f"Error: Failed to read file: {str(e)}"


@mcp.tool()
async def get_skill_info(skill_name: str) -> str:
    """
    Retrieves the documentation (SKILL.md) for a specific skill.

    Args:
        skill_name: The name of the skill (e.g., 'pdf-text-replace', 'image-crop-rotate')

    Returns:
        The content of the skill's SKILL.md file with usage instructions and examples.
    """
    content, skill_type, error = await _read_skill_file(skill_name, "SKILL.md")

    if error:
        return error

    # Add header with skill type
    header = f"Skill: {skill_name} ({skill_type})\n"
    header += f"Location: /app/uploads/skills/{skill_type}/{skill_name}/\n"
    header += "=" * 80 + "\n\n"

    return header + content


@mcp.tool()
async def get_skill_file(skill_name: str, filename: str) -> str:
    """
    Retrieves any markdown file from a skill's directory.
    This is useful when SKILL.md references other documentation files like EXAMPLES.md, API.md, etc.

    Args:
        skill_name: The name of the skill (e.g., 'pdf-text-replace', 'image-crop-rotate')
        filename: The name of the markdown file to read (e.g., 'EXAMPLES.md', 'API.md', 'README.md')

    Returns:
        The content of the requested file with /mnt/user-data paths replaced with /app/uploads.

    Example:
        get_skill_file('pdf-text-replace', 'EXAMPLES.md')
    """
    content, skill_type, error = await _read_skill_file(skill_name, filename)

    if error:
        return error

    # Add header with file info
    header = f"Skill: {skill_name} ({skill_type})\n"
    header += f"File: {filename}\n"
    header += f"Location: /app/uploads/skills/{skill_type}/{skill_name}/{filename}\n"
    header += "=" * 80 + "\n\n"

    return header + content


# --- REST API ENDPOINTS FOR SANDBOX CLIENT COMPATIBILITY ---
# These endpoints provide REST API access compatible with the instavm SDK client
# allowing local execution without cloud API

class MockContext:
    """Mock context for REST API calls that don't have MCP context"""
    async def report_progress(self, progress: int, message: str):
        # Log progress instead of reporting through MCP
        logger.info(f"Progress {progress}%: {message}")


# Use the streamable_http_app as it's the modern standard
app = mcp.streamable_http_app()

# Add custom REST API endpoints compatible with instavm SDK client
async def api_execute(request: Request):
    """
    REST API endpoint for executing Python code (compatible with InstaVM SDK).

    Request body (JSON):
        {
            "command": "print('hello world')",
            "session_id": "optional-ignored-for-local",
            "language": "python",  // optional, only python supported
            "timeout": 300  // optional, not used in local execution
        }

    Response (JSON):
        {
            "output": "hello world\\n",
            "status": "success"
        }
    or
        {
            "output": "",
            "error": "error message",
            "status": "error"
        }
    """
    try:
        # Parse request body
        body = await request.json()
        command = body.get("command")

        if not command:
            return {
                "output": "",
                "error": "Missing 'command' field in request body",
                "status": "error"
            }

        # Create mock context for progress reporting
        ctx = MockContext()

        # Execute the code
        result = await execute_python_code(command, ctx)

        # Check if result contains an error
        if result.startswith("Error:"):
            return JSONResponse({
                "output": "",
                "error": result,
                "status": "error"
            })

        return JSONResponse({
            "output": result,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error in /execute endpoint: {e}", exc_info=True)
        return JSONResponse({
            "output": "",
            "error": f"Error: {str(e)}",
            "status": "error"
        })


async def api_browser_navigate(request: Request):
    """
    REST API endpoint for browser navigation (compatible with InstaVM SDK).

    Request body (JSON):
        {
            "url": "https://example.com",
            "session_id": "optional-ignored-for-local",
            "wait_timeout": 30000  // optional
        }

    Response (JSON):
        {
            "status": "success",
            "url": "https://example.com",
            "title": "Example Domain"
        }
    or
        {
            "status": "error",
            "error": "error message"
        }
    """
    try:
        # Parse request body
        body = await request.json()
        url = body.get("url")

        if not url:
            return JSONResponse({
                "status": "error",
                "error": "Missing 'url' field in request body"
            })

        # Navigate and get text
        result = await navigate_and_get_all_visible_text(url)

        # Check if result contains an error
        if result.startswith("Error:"):
            return JSONResponse({
                "status": "error",
                "error": result
            })

        return JSONResponse({
            "status": "success",
            "url": url,
            "content": result,
            "title": "Navigation successful"
        })

    except Exception as e:
        logger.error(f"Error in /v1/browser/interactions/navigate endpoint: {e}", exc_info=True)
        return JSONResponse({
            "status": "error",
            "error": f"Error: {str(e)}"
        })


async def api_browser_extract_content(request: Request):
    """
    REST API endpoint for extracting browser content (compatible with InstaVM SDK).

    Request body (JSON):
        {
            "session_id": "optional-ignored-for-local",
            "url": "https://example.com",  // required for local execution
            "include_interactive": true,
            "include_anchors": true,
            "max_anchors": 50
        }

    Response (JSON):
        {
            "readable_content": {"content": "text content"},
            "status": "success"
        }
    """
    try:
        # Parse request body
        body = await request.json()
        url = body.get("url")

        if not url:
            return JSONResponse({
                "status": "error",
                "error": "Missing 'url' field in request body (required for local execution)"
            })

        # Navigate and get text
        result = await navigate_and_get_all_visible_text(url)

        # Check if result contains an error
        if result.startswith("Error:"):
            return JSONResponse({
                "status": "error",
                "error": result
            })

        return JSONResponse({
            "readable_content": {
                "content": result
            },
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error in /v1/browser/interactions/content endpoint: {e}", exc_info=True)
        return JSONResponse({
            "status": "error",
            "error": f"Error: {str(e)}"
        })

# Add routes to the Starlette app
app.add_route("/execute", api_execute, methods=["POST"])
app.add_route("/v1/browser/interactions/navigate", api_browser_navigate, methods=["POST"])
app.add_route("/v1/browser/interactions/content", api_browser_extract_content, methods=["POST"])