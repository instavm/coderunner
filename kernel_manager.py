
import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

import aiofiles
import websockets
import httpx

from utils import create_jupyter_request

logger = logging.getLogger(__name__)

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
    def __init__(self, kernel_id_file_path):
        self.kernels: Dict[str, KernelInfo] = {}
        self.lock = asyncio.Lock()
        self.busy_kernels: Set[str] = set()
        self._initialized = False
        self._health_check_task: Optional[asyncio.Task] = None
        self.kernel_id_file_path = kernel_id_file_path

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
            async with aiofiles.open(self.kernel_id_file_path, mode='r') as f:
                kernel_id = (await f.read()).strip()
                if kernel_id and await self._check_kernel_health(kernel_id):
                    return kernel_id
        except FileNotFoundError:
            # This is a normal case if the server is starting for the first time.
            pass
        except Exception as e:
            logger.warning(f"Could not read or validate existing kernel from {self.kernel_id_file_path}: {e}")
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
