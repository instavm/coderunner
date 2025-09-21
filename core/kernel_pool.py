"""Kernel pool management for CodeRunner execution contexts"""

import asyncio
import json
import logging
import pathlib
import time
import uuid
import websockets
from typing import Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

import httpx

logger = logging.getLogger(__name__)

# Configuration
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

# Directory configuration
SHARED_DIR = pathlib.Path("/app/uploads")
SHARED_DIR.mkdir(exist_ok=True)
KERNEL_ID_FILE_PATH = SHARED_DIR / "python_kernel_id.txt"


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
                    # Mark as busy and update usage time
                    self.busy_kernels.add(kernel_id)
                    kernel_info.state = KernelState.BUSY
                    kernel_info.last_used = datetime.now()
                    logger.debug(f"Allocated kernel {kernel_id}")
                    return kernel_id

            # No available kernels, try to create one if under limit
            if len(self.kernels) < MAX_KERNELS:
                new_kernel_id = await self._create_new_kernel()
                if new_kernel_id:
                    self.kernels[new_kernel_id] = KernelInfo(kernel_id=new_kernel_id)
                    self.busy_kernels.add(new_kernel_id)
                    self.kernels[new_kernel_id].state = KernelState.BUSY
                    logger.info(f"Created and allocated new kernel: {new_kernel_id}")
                    return new_kernel_id

            logger.warning("No kernels available and cannot create new ones")
            return None

    async def release_kernel(self, kernel_id: str, failed: bool = False):
        """Release a kernel back to the pool"""
        async with self.lock:
            if kernel_id in self.busy_kernels:
                self.busy_kernels.remove(kernel_id)

            if kernel_id in self.kernels:
                if failed:
                    self.kernels[kernel_id].failure_count += 1
                    self.kernels[kernel_id].state = KernelState.FAILED
                    logger.warning(f"Kernel {kernel_id} marked as failed")

                    # Remove kernel if it has too many failures
                    if self.kernels[kernel_id].failure_count >= 3:
                        await self._remove_kernel(kernel_id)
                        logger.info(f"Removed kernel {kernel_id} due to repeated failures")
                else:
                    self.kernels[kernel_id].state = KernelState.HEALTHY
                    self.kernels[kernel_id].failure_count = 0
                    logger.debug(f"Released kernel {kernel_id}")

    async def _get_existing_kernel(self) -> Optional[str]:
        """Get existing kernel ID from file"""
        try:
            if KERNEL_ID_FILE_PATH.exists():
                with open(KERNEL_ID_FILE_PATH, 'r') as f:
                    kernel_id = f.read().strip()
                if kernel_id:
                    # Verify kernel is still alive
                    if await self._check_kernel_health(kernel_id):
                        return kernel_id
                    else:
                        logger.warning(f"Existing kernel {kernel_id} is not responsive")
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
                response.raise_for_status()
                kernel_data = response.json()
                kernel_id = kernel_data["id"]
                logger.info(f"Created new kernel: {kernel_id}")
                return kernel_id
        except Exception as e:
            logger.error(f"Failed to create new kernel: {e}")
            return None

    async def _remove_kernel(self, kernel_id: str):
        """Remove a kernel from Jupyter and pool"""
        try:
            async with httpx.AsyncClient() as client:
                await client.delete(
                    f"{JUPYTER_HTTP_URL}/api/kernels/{kernel_id}",
                    timeout=10.0
                )
            logger.info(f"Removed kernel from Jupyter: {kernel_id}")
        except Exception as e:
            logger.warning(f"Failed to remove kernel from Jupyter: {e}")

        # Remove from pool
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
                ping_timeout=WEBSOCKET_PING_TIMEOUT,
                close_timeout=5
            ) as jupyter_ws:
                # Send a simple execution request
                msg_id = str(uuid.uuid4())
                request = {
                    "header": {
                        "msg_id": msg_id,
                        "msg_type": "execute_request",
                        "version": "5.3"
                    },
                    "parent_header": {},
                    "metadata": {},
                    "content": {
                        "code": "1+1",
                        "silent": True,
                        "store_history": False
                    }
                }

                await jupyter_ws.send(json.dumps(request))

                # Wait for status response
                timeout_time = time.time() + 10
                while time.time() < timeout_time:
                    try:
                        message_str = await asyncio.wait_for(jupyter_ws.recv(), timeout=2.0)
                        message_data = json.loads(message_str)
                        if (message_data.get("parent_header", {}).get("msg_id") == msg_id and
                            message_data.get("header", {}).get("msg_type") == "status" and
                            message_data.get("content", {}).get("execution_state") == "idle"):
                            return True
                    except asyncio.TimeoutError:
                        continue

                return False
        except Exception as e:
            logger.debug(f"Health check failed for kernel {kernel_id}: {e}")
            return False

    async def _health_check_loop(self):
        """Background task to periodically check kernel health"""
        while True:
            try:
                await asyncio.sleep(KERNEL_HEALTH_CHECK_INTERVAL)
                
                async with self.lock:
                    kernels_to_check = [
                        (kernel_id, kernel_info) 
                        for kernel_id, kernel_info in self.kernels.items()
                        if kernel_info.needs_health_check() and kernel_id not in self.busy_kernels
                    ]

                # Check health outside of lock to avoid blocking
                for kernel_id, kernel_info in kernels_to_check:
                    is_healthy = await self._check_kernel_health(kernel_id)
                    
                    async with self.lock:
                        if kernel_id in self.kernels:  # Kernel might have been removed
                            self.kernels[kernel_id].last_health_check = datetime.now()
                            if not is_healthy:
                                self.kernels[kernel_id].state = KernelState.UNRESPONSIVE
                                logger.warning(f"Kernel {kernel_id} is unresponsive")
                            elif self.kernels[kernel_id].state == KernelState.UNRESPONSIVE:
                                self.kernels[kernel_id].state = KernelState.HEALTHY
                                logger.info(f"Kernel {kernel_id} recovered")

            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

    async def execute_on_kernel(self, kernel_id: str, command: str) -> dict:
        """Execute code on specific kernel for session management"""
        jupyter_ws_url = f"{JUPYTER_WS_URL}/api/kernels/{kernel_id}/channels"
        final_output_lines = []
        error_output = []
        sent_msg_id = None

        try:
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
                
                while not execution_complete and (time.time() - start_time) < WEBSOCKET_TIMEOUT:
                    try:
                        message_str = await asyncio.wait_for(jupyter_ws.recv(), timeout=30.0)
                    except asyncio.TimeoutError:
                        continue

                    try:
                        message_data = json.loads(message_str)
                    except json.JSONDecodeError:
                        continue

                    parent_msg_id = message_data.get("parent_header", {}).get("msg_id")
                    if parent_msg_id != sent_msg_id:
                        continue

                    msg_type = message_data.get("header", {}).get("msg_type")
                    content = message_data.get("content", {})

                    if msg_type == "stream":
                        stream_text = content.get("text", "")
                        final_output_lines.append(stream_text)

                    elif msg_type in ["execute_result", "display_data"]:
                        result_text = content.get("data", {}).get("text/plain", "")
                        final_output_lines.append(result_text)

                    elif msg_type == "error":
                        error_traceback = "\n".join(content.get("traceback", []))
                        error_output.append(error_traceback)

                    elif msg_type == "status" and content.get("execution_state") == "idle":
                        execution_complete = True

                if not execution_complete:
                    elapsed = time.time() - start_time
                    raise KernelTimeoutError(f"Execution timed out after {elapsed:.0f} seconds")

                execution_time = time.time() - start_time
                
                return {
                    "stdout": "".join(final_output_lines),
                    "stderr": "\n".join(error_output),
                    "execution_time": execution_time,
                    "success": len(error_output) == 0
                }

        except Exception as e:
            logger.error(f"Error executing on kernel {kernel_id}: {e}")
            return {
                "stdout": "",
                "stderr": str(e),
                "execution_time": 0.0,
                "success": False
            }


def create_jupyter_request(code: str) -> tuple[str, str]:
    """Create a Jupyter execute_request message"""
    msg_id = str(uuid.uuid4())
    
    request = {
        "header": {
            "msg_id": msg_id,
            "msg_type": "execute_request",
            "version": "5.3"
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True
        }
    }
    
    return msg_id, json.dumps(request)


# Global kernel pool instance
kernel_pool = KernelPool()