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
import subprocess
from typing import Dict, Optional, Set, List, Any
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

# --- UNIVERSAL NPX MCP BRIDGE ---

@dataclass
class MCPServer:
    """Represents a stdio-based MCP server"""
    name: str
    command: List[str]
    process: Optional[subprocess.Popen] = None
    tools: List[Dict[str, Any]] = field(default_factory=list)
    resources: List[Dict[str, Any]] = field(default_factory=list)
    prompts: List[Dict[str, Any]] = field(default_factory=list)

class UniversalMCPBridge:
    """
    Universal bridge to expose any stdio-based MCP server via HTTP.

    This class allows you to run any NPX-based MCP server and expose its tools
    through your existing FastMCP HTTP endpoint.

    Example:
        bridge = UniversalMCPBridge()
        await bridge.add_server("filesystem", ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/path"])
        result = await bridge.call_tool("filesystem", "read_file", {"path": "/path/file.txt"})
    """

    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.reader_tasks: Dict[str, asyncio.Task] = {}

    async def add_server(self, name: str, command: List[str]) -> bool:
        """
        Add and initialize an MCP server.

        Args:
            name: Unique name for this server
            command: Command array (e.g., ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/path"])

        Returns:
            True if server started successfully
        """
        try:
            logger.info(f"Starting MCP server '{name}' with command: {' '.join(command)}")

            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            server = MCPServer(name=name, command=command, process=process)
            self.servers[name] = server

            # Start reader task for this server
            self.reader_tasks[name] = asyncio.create_task(self._read_responses(name))

            # Initialize the server
            init_response = await self._send_request(name, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "coderunner-bridge", "version": "1.0.0"}
            })

            if "error" in init_response:
                logger.error(f"Failed to initialize server '{name}': {init_response['error']}")
                return False

            # Send initialized notification
            await self._send_notification(name, "notifications/initialized")

            # Discover capabilities
            await self._discover_capabilities(name)

            logger.info(f"Successfully initialized MCP server '{name}' with {len(server.tools)} tools")
            return True

        except Exception as e:
            logger.error(f"Failed to start MCP server '{name}': {e}")
            return False

    async def _read_responses(self, server_name: str):
        """Background task to read responses from a server's stdout"""
        server = self.servers.get(server_name)
        if not server or not server.process:
            return

        loop = asyncio.get_event_loop()

        while server.process.poll() is None:
            try:
                # Read line from stdout in executor to avoid blocking
                line = await loop.run_in_executor(None, server.process.stdout.readline)

                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    response = json.loads(line)
                    msg_id = response.get("id")

                    # Handle response to a request
                    if msg_id and msg_id in self.pending_requests:
                        future = self.pending_requests.pop(msg_id)
                        if not future.done():
                            future.set_result(response)

                    # Handle notifications (no id field)
                    elif "method" in response:
                        logger.debug(f"Received notification from '{server_name}': {response.get('method')}")

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from '{server_name}': {line[:100]}")

            except Exception as e:
                logger.error(f"Error reading from '{server_name}': {e}")
                break

    async def _send_request(self, server_name: str, method: str, params: Any) -> Dict:
        """Send a JSON-RPC request to a server"""
        server = self.servers.get(server_name)
        if not server or not server.process:
            return {"error": f"Server '{server_name}' not found or not running"}

        msg_id = str(uuid.uuid4())
        request = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
            "params": params
        }

        # Create future for response
        future = asyncio.Future()
        self.pending_requests[msg_id] = future

        # Send request
        try:
            request_json = json.dumps(request) + "\n"
            server.process.stdin.write(request_json)
            server.process.stdin.flush()

            # Wait for response with timeout
            response = await asyncio.wait_for(future, timeout=60.0)
            return response

        except asyncio.TimeoutError:
            self.pending_requests.pop(msg_id, None)
            return {"error": f"Request timeout for method '{method}'"}
        except Exception as e:
            self.pending_requests.pop(msg_id, None)
            return {"error": f"Request failed: {str(e)}"}

    async def _send_notification(self, server_name: str, method: str, params: Any = None):
        """Send a JSON-RPC notification (no response expected)"""
        server = self.servers.get(server_name)
        if not server or not server.process:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            notification["params"] = params

        try:
            notification_json = json.dumps(notification) + "\n"
            server.process.stdin.write(notification_json)
            server.process.stdin.flush()
        except Exception as e:
            logger.error(f"Failed to send notification to '{server_name}': {e}")

    async def _discover_capabilities(self, server_name: str):
        """Discover tools, resources, and prompts from a server"""
        server = self.servers.get(server_name)
        if not server:
            return

        # List tools
        tools_response = await self._send_request(server_name, "tools/list", {})
        if "result" in tools_response:
            server.tools = tools_response["result"].get("tools", [])
            logger.info(f"Discovered {len(server.tools)} tools from '{server_name}'")

        # List resources (optional)
        resources_response = await self._send_request(server_name, "resources/list", {})
        if "result" in resources_response:
            server.resources = resources_response["result"].get("resources", [])
            logger.info(f"Discovered {len(server.resources)} resources from '{server_name}'")

        # List prompts (optional)
        prompts_response = await self._send_request(server_name, "prompts/list", {})
        if "result" in prompts_response:
            server.prompts = prompts_response["result"].get("prompts", [])
            logger.info(f"Discovered {len(server.prompts)} prompts from '{server_name}'")

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on a specific server.

        Args:
            server_name: Name of the server
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result or error
        """
        response = await self._send_request(server_name, "tools/call", {
            "name": tool_name,
            "arguments": arguments
        })

        if "error" in response:
            return {"error": response["error"]}

        return response.get("result", {})

    def list_all_tools(self) -> Dict[str, List[Dict]]:
        """List all tools from all servers"""
        all_tools = {}
        for name, server in self.servers.items():
            all_tools[name] = server.tools or []
        return all_tools

    async def shutdown(self):
        """Shutdown all servers"""
        for server in self.servers.values():
            if server.process:
                server.process.terminate()
                try:
                    server.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.process.kill()

        # Cancel all reader tasks
        for task in self.reader_tasks.values():
            task.cancel()

# Global bridge instance
mcp_bridge = UniversalMCPBridge()

# --- MCP BRIDGE CONFIGURATION ---

# Configuration for NPX-based MCP servers to expose via HTTP
# Add any NPX MCP server here with its command and optional environment variables
MCP_BRIDGE_SERVERS = [
    {
        "name": "filesystem",
        "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", str(SHARED_DIR)],
        "description": "File system operations (read, write, list, etc.)",
        "enabled": True
    },
    {
        "name": "chrome-devtools",
        "command": ["npx", "-y", "chrome-devtools-mcp@latest", "--headless=true"],
        "description": "Chrome DevTools Protocol for browser automation and debugging (headless mode)",
        "enabled": True
    },
    # Uncomment and configure as needed:
    # {
    #     "name": "github",
    #     "command": ["npx", "-y", "@modelcontextprotocol/server-github"],
    #     "description": "GitHub repository operations",
    #     "enabled": bool(os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"))  # Only enable if token is set
    # },
    # {
    #     "name": "puppeteer",
    #     "command": ["npx", "-y", "@modelcontextprotocol/server-puppeteer"],
    #     "description": "Web scraping and browser automation",
    #     "enabled": True
    # },
    # {
    #     "name": "memory",
    #     "command": ["npx", "-y", "@modelcontextprotocol/server-memory"],
    #     "description": "Persistent memory across conversations",
    #     "enabled": True
    # },
    # {
    #     "name": "brave-search",
    #     "command": ["npx", "-y", "@modelcontextprotocol/server-brave-search"],
    #     "description": "Web search via Brave Search API",
    #     "enabled": bool(os.getenv("BRAVE_API_KEY"))
    # },
]

async def initialize_mcp_bridges():
    """Initialize all enabled NPX-based MCP servers"""
    logger.info("Initializing MCP bridge servers...")

    initialized_count = 0
    for config in MCP_BRIDGE_SERVERS:
        if not config.get("enabled", True):
            logger.info(f"Skipping disabled server: {config['name']}")
            continue

        success = await mcp_bridge.add_server(config["name"], config["command"])
        if success:
            logger.info(f"✅ Loaded MCP server: {config['name']} - {config.get('description', '')}")
            initialized_count += 1
        else:
            logger.error(f"❌ Failed to load MCP server: {config['name']}")

    logger.info(f"MCP bridge initialization complete: {initialized_count}/{len([c for c in MCP_BRIDGE_SERVERS if c.get('enabled', True)])} servers loaded")
    return initialized_count

def create_bridge_tool_wrapper(server_name: str, tool_name: str, tool_description: str):
    """
    Create a wrapper function for a bridged MCP tool.

    This dynamically creates a function that can be registered with FastMCP
    to expose tools from NPX-based MCP servers.
    """
    async def wrapper(**kwargs):
        """Dynamically created wrapper for bridged MCP tool"""
        try:
            result = await mcp_bridge.call_tool(server_name, tool_name, kwargs)

            if "error" in result:
                return f"Error: {result['error']}"

            # Extract text content from MCP response
            content = result.get("content", [])
            if isinstance(content, list) and len(content) > 0:
                first_item = content[0]
                if isinstance(first_item, dict):
                    return first_item.get("text", str(result))
            return str(result)

        except Exception as e:
            logger.error(f"Error calling bridged tool {server_name}.{tool_name}: {e}")
            return f"Error: {str(e)}"

    # Set function metadata for FastMCP
    wrapper.__name__ = f"{server_name}_{tool_name}"
    wrapper.__doc__ = f"[{server_name}] {tool_description}"

    return wrapper

async def register_bridge_tools():
    """
    Dynamically register all tools from all bridged MCP servers.

    This function discovers all tools from the bridged servers and registers
    them as FastMCP tools, making them available via the HTTP endpoint.
    """
    logger.info("Registering bridged MCP tools...")

    all_tools = mcp_bridge.list_all_tools()
    total_tools = 0

    for server_name, tools in all_tools.items():
        for tool in tools:
            tool_name = tool["name"]
            tool_description = tool.get("description", f"Tool from {server_name} server")

            # Create wrapper function
            wrapper = create_bridge_tool_wrapper(server_name, tool_name, tool_description)

            # Register with FastMCP
            mcp.tool()(wrapper)

            logger.info(f"Registered: {server_name}_{tool_name}")
            total_tools += 1

    logger.info(f"Successfully registered {total_tools} bridged tools from {len(all_tools)} servers")
    return total_tools

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


# --- STARTUP INITIALIZATION ---

_bridge_initialized = False

async def ensure_bridges_initialized():
    """
    Ensure MCP bridges are initialized (lazy initialization).
    This is called on first request to avoid blocking module import.
    """
    global _bridge_initialized

    if _bridge_initialized:
        return

    logger.info("Starting MCP bridge initialization...")

    try:
        bridge_count = await initialize_mcp_bridges()
        if bridge_count > 0:
            # Register all discovered tools
            tool_count = await register_bridge_tools()
            logger.info(f"MCP bridges ready: {bridge_count} servers, {tool_count} tools")
        else:
            logger.info("No MCP bridge servers configured or enabled")
    except Exception as e:
        logger.error(f"Failed to initialize MCP bridges: {e}")
        # Continue even if bridges fail - the main functionality should still work

    _bridge_initialized = True
    logger.info("MCP bridge initialization complete!")


# Add a tool to manually trigger bridge initialization
@mcp.tool()
async def initialize_bridges() -> str:
    """
    Initialize all configured NPX-based MCP server bridges.
    This is called automatically on first use, but can be called manually if needed.

    Returns:
        Status message indicating how many servers and tools were initialized.
    """
    global _bridge_initialized

    if _bridge_initialized:
        all_tools = mcp_bridge.list_all_tools()
        total_tools = sum(len(tools) for tools in all_tools.values())
        return f"MCP bridges already initialized: {len(all_tools)} servers, {total_tools} tools"

    await ensure_bridges_initialized()

    all_tools = mcp_bridge.list_all_tools()
    total_tools = sum(len(tools) for tools in all_tools.values())
    return f"MCP bridges initialized successfully: {len(all_tools)} servers, {total_tools} tools"


@mcp.tool()
async def list_bridged_tools() -> str:
    """
    List all available tools from bridged NPX MCP servers.

    Returns:
        Formatted list of all bridged MCP servers and their tools.
    """
    if not _bridge_initialized:
        await ensure_bridges_initialized()

    all_tools = mcp_bridge.list_all_tools()

    if not all_tools:
        return "No MCP bridge servers configured. Edit MCP_BRIDGE_SERVERS in server.py to add NPX-based MCP servers."

    result = "Bridged MCP Servers and Tools:\n\n"

    for server_name, tools in all_tools.items():
        # Find server description from config
        server_desc = next(
            (s.get("description", "") for s in MCP_BRIDGE_SERVERS if s["name"] == server_name),
            ""
        )

        result += f"📦 {server_name}"
        if server_desc:
            result += f" - {server_desc}"
        result += f"\n  {len(tools)} tool(s):\n"

        for tool in tools:
            tool_name = tool["name"]
            tool_desc = tool.get("description", "No description")
            result += f"  • {server_name}_{tool_name}: {tool_desc}\n"

        result += "\n"

    return result


# Use the streamable_http_app as it's the modern standard
app = mcp.streamable_http_app()


# Add lifespan handler to initialize on startup
@app.on_event("startup")
async def startup_event():
    """Initialize services when the app starts"""
    logger.info("CodeRunner MCP Server starting up...")
    await ensure_bridges_initialized()
    logger.info("CodeRunner MCP Server ready!")