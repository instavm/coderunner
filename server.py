# --- IMPORTS ---
import asyncio
import base64
import binascii
import json
import logging
import os
import re
import pathlib
import shutil
import stat
import time
import uuid
import zipfile
from typing import Dict, Optional, Set, TypedDict
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from urllib.parse import urlsplit

import aiofiles
import websockets
import httpx
# Import Context for progress reporting
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.transport_security import TransportSecuritySettings
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
# Extra hostnames (comma-separated) that may be used to reach this server,
# e.g. a LAN IP or a custom DNS name.
EXTRA_ALLOWED_HOSTNAMES = [
    h.strip().lower()
    for h in os.environ.get("CODERUNNER_EXTRA_HOSTS", "").split(",")
    if h.strip()
]

# Configure DNS rebinding protection to allow coderunner.local
mcp = FastMCP(
    "CodeRunner",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "localhost:*",
            "127.0.0.1:*",
            "coderunner.local:*",
            "0.0.0.0:*",
            *[f"{h}:*" for h in EXTRA_ALLOWED_HOSTNAMES],
        ],
        allowed_origins=[
            "http://localhost:*",
            "http://127.0.0.1:*",
            "http://coderunner.local:*",
            *[f"http://{h}:*" for h in EXTRA_ALLOWED_HOSTNAMES],
        ],
    )
)

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

class SessionConflictError(ValueError):
    """Raised when a requested session ID already exists"""
    pass


# --- KERNEL MANAGEMENT CLASSES ---

class KernelState(Enum):
    HEALTHY = "healthy"
    BUSY = "busy"
    UNRESPONSIVE = "unresponsive"

@dataclass
class KernelInfo:
    kernel_id: str
    state: KernelState = KernelState.HEALTHY
    last_used: datetime = field(default_factory=datetime.now)
    last_health_check: datetime = field(default_factory=datetime.now)
    current_operation: Optional[str] = None

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
            if self._initialized:
                return
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

    async def release_kernel(self, kernel_id: str):
        """Release a kernel back to the pool"""
        async with self.lock:
            if kernel_id in self.busy_kernels:
                self.busy_kernels.remove(kernel_id)

            if kernel_id in self.kernels:
                kernel_info = self.kernels[kernel_id]
                kernel_info.state = KernelState.HEALTHY
                kernel_info.current_operation = None
                logger.info(f"Released kernel {kernel_id} back to pool")

    async def discard_kernel(self, kernel_id: str):
        """Shutdown a reserved kernel and replenish the warm pool."""
        async with self.lock:
            self.kernels.pop(kernel_id, None)
            self.busy_kernels.discard(kernel_id)
            needs_replacement = len(self.kernels) < MIN_KERNELS

        await self._shutdown_kernel(kernel_id)
        if needs_replacement:
            new_kernel_id = await self._create_new_kernel()
            if new_kernel_id:
                keep_kernel = False
                async with self.lock:
                    if len(self.kernels) < MIN_KERNELS:
                        keep_kernel = True
                        self.kernels[new_kernel_id] = KernelInfo(kernel_id=new_kernel_id)
                if not keep_kernel:
                    await self._shutdown_kernel(new_kernel_id)

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
                    if await self._check_kernel_health(kernel_id):
                        logger.info(f"Created new kernel: {kernel_id}")
                        return kernel_id
                    await client.delete(
                        f"{JUPYTER_HTTP_URL}/api/kernels/{kernel_id}",
                        timeout=10.0,
                    )
                    logger.error(f"New kernel did not become ready: {kernel_id}")
                else:
                    logger.error(f"Failed to create kernel: {response.status_code}")
        except Exception as e:
            logger.error(f"Error creating kernel: {e}")
        return None

    async def _remove_kernel(self, kernel_id: str):
        """Remove and shutdown a kernel"""
        await self._shutdown_kernel(kernel_id)
        self.kernels.pop(kernel_id, None)
        self.busy_kernels.discard(kernel_id)

    async def _shutdown_kernel(self, kernel_id: str):
        """Shutdown a Jupyter kernel without modifying pool state."""
        try:
            async with httpx.AsyncClient() as client:
                await client.delete(
                    f"{JUPYTER_HTTP_URL}/api/kernels/{kernel_id}",
                    timeout=10.0
                )
            logger.info(f"Removed kernel: {kernel_id}")
        except Exception as e:
            logger.warning(f"Error removing kernel {kernel_id}: {e}")

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


class PythonSessionInfo(TypedDict):
    session_id: str
    status: str
    created_at: str


@dataclass
class PythonSession:
    session_id: str
    kernel_id: str
    created_at: datetime = field(default_factory=datetime.now)
    execution_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closed: bool = False

    def info(self, status: str = "active") -> PythonSessionInfo:
        return {
            "session_id": self.session_id,
            "status": status,
            "created_at": self.created_at.isoformat(),
        }


class PythonSessionManager:
    def __init__(self):
        self.sessions: Dict[str, PythonSession] = {}
        self.lock = asyncio.Lock()

    async def start(self, requested_id: Optional[str] = None) -> PythonSessionInfo:
        await kernel_pool.initialize()
        if requested_id is not None and not isinstance(requested_id, str):
            raise ValueError("Session ID must be a string.")
        session_id = requested_id or f"session_{uuid.uuid4().hex[:12]}"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", session_id):
            raise ValueError(
                "Session IDs must be 1-64 characters, start with a letter or number, "
                "and contain only letters, numbers, dots, dashes, or underscores."
            )

        async with self.lock:
            if session_id in self.sessions:
                raise SessionConflictError(f"Session '{session_id}' already exists.")
            kernel_id = await kernel_pool.get_available_kernel()
            if not kernel_id:
                raise NoKernelAvailableError(f"Maximum of {MAX_KERNELS} active sessions reached.")
            session = PythonSession(session_id=session_id, kernel_id=kernel_id)
            self.sessions[session_id] = session
            logger.info(f"Started Python session {session_id} on kernel {kernel_id}")
            return session.info()

    async def execute(self, session_id: str, command: str, ctx: Context) -> str:
        async with self.lock:
            session = self.sessions.get(session_id)
        if not session:
            return f"Error: Session '{session_id}' not found."

        async with session.execution_lock:
            if session.closed:
                return f"Error: Session '{session_id}' is closed."
            try:
                return await _execute_on_kernel(session.kernel_id, command, ctx)
            except KernelExecutionError as exc:
                return f"Error: {exc}"
            except Exception as exc:
                async with self.lock:
                    self.sessions.pop(session_id, None)
                    session.closed = True
                await kernel_pool.discard_kernel(session.kernel_id)
                return f"Error: Session '{session_id}' failed and was closed: {exc}"

    async def stop(self, session_id: str) -> PythonSessionInfo:
        async with self.lock:
            session = self.sessions.pop(session_id, None)
            if session:
                session.closed = True
        if not session:
            raise ValueError(f"Session '{session_id}' not found.")

        async with session.execution_lock:
            await kernel_pool.discard_kernel(session.kernel_id)
        logger.info(f"Stopped Python session {session_id}")
        return session.info(status="stopped")

    async def get(self, session_id: str) -> Optional[PythonSessionInfo]:
        async with self.lock:
            session = self.sessions.get(session_id)
            return session.info() if session else None

    async def list(self) -> list[PythonSessionInfo]:
        async with self.lock:
            return [self.sessions[key].info() for key in sorted(self.sessions)]


python_sessions = PythonSessionManager()



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
                await kernel_pool.release_kernel(kernel_id)
                return result
            except KernelExecutionError as e:
                await kernel_pool.release_kernel(kernel_id)
                return f"Error: {e}"
            except Exception as e:
                await kernel_pool.discard_kernel(kernel_id)
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
async def execute_python_code(
    command: str,
    ctx: Context,
    session_id: Optional[str] = None,
) -> str:
    """
    Executes a string of Python code in a persistent Jupyter kernel and returns the final output.
    Uses kernel pool management with automatic retry and recovery for long-running operations.
    Streams intermediate output (stdout) as progress updates.

    Args:
        command: The Python code to execute as a single string.
        ctx: The MCP Context object, used for reporting progress.
        session_id: Optional named session created by start_python_session.
    """
    try:
        # Initialize kernel pool if not already done
        if not kernel_pool._initialized:
            await ctx.report_progress(progress=10, message="Initializing kernel pool...")
            await kernel_pool.initialize()

        if session_id:
            result = await python_sessions.execute(session_id, command, ctx)
        else:
            result = await execute_with_retry(command, ctx)
        return result

    except Exception as e:
        logger.error(f"Fatal error in execute_python_code: {e}", exc_info=True)
        return f"Error: Failed to execute code: {str(e)}"


@mcp.tool()
async def start_python_session(session_id: Optional[str] = None) -> PythonSessionInfo:
    """Start a named Python session with an isolated persistent kernel."""
    return await python_sessions.start(session_id)


@mcp.tool()
async def list_python_sessions() -> list[PythonSessionInfo]:
    """List active named Python sessions."""
    return await python_sessions.list()


@mcp.tool()
async def stop_python_session(session_id: str) -> PythonSessionInfo:
    """Stop a named Python session and discard its kernel state."""
    return await python_sessions.stop(session_id)


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


def _extract_skill_archive(archive_path: pathlib.Path) -> None:
    destination = USER_SKILLS_DIR.resolve()
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe path in skill archive: {member.filename}")

            mode = member.external_attr >> 16
            file_type = stat.S_IFMT(mode)
            if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f"Unsupported file type in skill archive: {member.filename}")

        for member in archive.infolist():
            target = destination / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                raise ValueError(f"Unsafe symlink target in skill archive: {member.filename}")
            with archive.open(member, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


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
                    _extract_skill_archive(item)
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
        skill_file_path = None
        skill_type = None

        # Resolve the requested path and make sure it stays inside the
        # skill directory, so that names like "../../etc" cannot escape it.
        for base_dir, category in ((PUBLIC_SKILLS_DIR, "public"), (USER_SKILLS_DIR, "user")):
            candidate = (base_dir / skill_name / filename).resolve()
            if not candidate.is_relative_to(base_dir.resolve()):
                return None, None, f"Error: Invalid skill or file name."
            if candidate.is_file():
                skill_file_path = candidate
                skill_type = category
                break

        if skill_file_path is None:
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

# Host/Origin validation for the plain REST routes below. FastMCP applies
# its transport security settings to /mcp only, so without this a web page
# could issue a drive-by POST to /execute from the user's browser.
ALLOWED_HOSTNAMES = {"localhost", "127.0.0.1", "coderunner.local", "0.0.0.0", *EXTRA_ALLOWED_HOSTNAMES}


def _header_hostname(value: str, *, origin: bool = False) -> Optional[str]:
    try:
        parsed = urlsplit(value if origin else f"//{value}")
        if origin and parsed.scheme.lower() not in {"http", "https"}:
            return None
        if parsed.username is not None or parsed.password is not None:
            return None
        if parsed.path or parsed.query or parsed.fragment:
            return None
        parsed.port
        return parsed.hostname.lower() if parsed.hostname else None
    except ValueError:
        return None


class HostOriginValidator:
    def __init__(self, asgi_app):
        self.asgi_app = asgi_app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and not self._is_allowed(scope):
            response = JSONResponse({"error": "Invalid Host or Origin header"}, status_code=403)
            await response(scope, receive, send)
            return
        await self.asgi_app(scope, receive, send)

    @staticmethod
    def _is_allowed(scope) -> bool:
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        host = _header_hostname(headers.get("host", ""))
        if host not in ALLOWED_HOSTNAMES:
            return False
        origin = headers.get("origin")
        if origin:
            origin_host = _header_hostname(origin, origin=True)
            if origin_host not in ALLOWED_HOSTNAMES:
                return False
        return True


async def api_health(request: Request):
    """Liveness endpoint used by the installer and container healthchecks."""
    return JSONResponse({"status": "ok"})

# Add custom REST API endpoints compatible with instavm SDK client
async def api_execute(request: Request):
    """
    REST API endpoint for executing Python code (compatible with InstaVM SDK).

    Request body (JSON):
        {
            "command": "print('hello world')",
            "session_id": "optional named Python session",
            "language": "python",  // optional, only python supported
            "timeout": 300  // optional, not used in local execution
        }

    Response (JSON) - matches api.instavm.io/execute format:
        {
            "stdout": "hello world\\n",
            "stderr": "",
            "execution_time": 0.39,
            "cpu_time": 0.03
        }
    """
    import time
    start_time = time.time()

    try:
        # Parse request body
        body = await request.json()

        # SDK sends "code" field, direct API calls use "command"
        command = body.get("code") or body.get("command")

        if not command:
            return JSONResponse({
                "stdout": "",
                "stderr": "Missing 'code' or 'command' field in request body",
                "execution_time": 0.0,
                "cpu_time": 0.0
            }, status_code=400)

        # Create mock context for progress reporting
        ctx = MockContext()

        # Execute the code
        result = await execute_python_code(command, ctx, session_id=body.get("session_id"))

        # Calculate execution time
        execution_time = time.time() - start_time

        # Check if result contains an error
        if result.startswith("Error:"):
            return JSONResponse({
                "stdout": "",
                "stderr": result,
                "execution_time": execution_time,
                "cpu_time": execution_time  # Approximate CPU time as execution time
            })

        # For compatibility with api.instavm.io, return stdout/stderr format
        # Since execute_python_code returns combined output, we put it all in stdout
        return JSONResponse({
            "stdout": result,
            "stderr": "",
            "execution_time": execution_time,
            "cpu_time": execution_time  # Approximate CPU time as execution time
        })

    except Exception as e:
        logger.error(f"Error in /execute endpoint: {e}", exc_info=True)
        execution_time = time.time() - start_time
        return JSONResponse({
            "stdout": "",
            "stderr": f"Error: {str(e)}",
            "execution_time": execution_time,
            "cpu_time": execution_time
        }, status_code=500)


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

# --- SESSION MANAGEMENT ENDPOINTS FOR SDK COMPATIBILITY ---

async def api_start_session(request: Request):
    """Start a named Python session (compatible with InstaVM SDK)."""
    try:
        raw_body = await request.body()
        body = json.loads(raw_body) if raw_body else {}
        info = await python_sessions.start(body.get("session_id"))
        return JSONResponse(info, status_code=201)
    except json.JSONDecodeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except SessionConflictError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except NoKernelAvailableError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


async def api_get_session(request: Request):
    """Get one session or list all active sessions."""
    session_id = request.query_params.get("session_id")
    if not session_id:
        return JSONResponse({"sessions": await python_sessions.list()})
    info = await python_sessions.get(session_id)
    if not info:
        return JSONResponse({"error": f"Session '{session_id}' not found."}, status_code=404)
    return JSONResponse(info)


async def api_stop_session(request: Request):
    """Stop a named Python session and discard its kernel."""
    try:
        raw_body = await request.body()
        body = json.loads(raw_body) if raw_body else {}
        session_id = request.query_params.get("session_id") or body.get("session_id")
        if not session_id:
            return JSONResponse({"error": "Missing session_id."}, status_code=400)
        return JSONResponse(await python_sessions.stop(session_id))
    except json.JSONDecodeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)


# Add routes to the Starlette app
app.add_route("/health", api_health, methods=["GET"])
app.add_route("/execute", api_execute, methods=["POST"])
app.add_route("/v1/sessions/session", api_start_session, methods=["POST"])
app.add_route("/v1/sessions/session", api_get_session, methods=["GET"])
app.add_route("/v1/sessions/session", api_stop_session, methods=["DELETE"])
app.add_route("/v1/browser/interactions/navigate", api_browser_navigate, methods=["POST"])
app.add_route("/v1/browser/interactions/content", api_browser_extract_content, methods=["POST"])

# Wrap the app last so validation covers every route
app = HostOriginValidator(app)
