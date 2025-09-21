import asyncio
import os
import pytest
from unittest.mock import MagicMock, AsyncMock

from kernel_manager import KernelPool, KernelState

# Set up a dummy kernel ID file
KERNEL_ID_FILE = "./test_kernel_id.txt"

@pytest.fixture
def kernel_pool():
    """Provides a KernelPool instance for testing."""
    # Clean up the dummy file before each test
    if os.path.exists(KERNEL_ID_FILE):
        os.remove(KERNEL_ID_FILE)

    pool = KernelPool(KERNEL_ID_FILE)
    pool._create_new_kernel = AsyncMock(return_value="test_kernel_id")
    pool._check_kernel_health = AsyncMock(return_value=True)
    pool._remove_kernel = AsyncMock()
    return pool

@pytest.mark.asyncio
async def test_initialize_creates_min_kernels(kernel_pool: KernelPool):
    """Tests that the kernel pool initializes with the minimum number of kernels."""
    await kernel_pool.initialize()
    assert len(kernel_pool.kernels) == kernel_pool.MIN_KERNELS
    assert kernel_pool._create_new_kernel.call_count == kernel_pool.MIN_KERNELS

@pytest.mark.asyncio
async def test_get_available_kernel_returns_kernel(kernel_pool: KernelPool):
    """Tests that get_available_kernel returns a kernel ID."""
    await kernel_pool.initialize()
    kernel_id = await kernel_pool.get_available_kernel()
    assert kernel_id is not None
    assert kernel_id in kernel_pool.kernels
    assert kernel_pool.kernels[kernel_id].state == KernelState.BUSY

@pytest.mark.asyncio
async def test_release_kernel_marks_kernel_as_healthy(kernel_pool: KernelPool):
    """Tests that release_kernel marks a kernel as healthy."""
    await kernel_pool.initialize()
    kernel_id = await kernel_pool.get_available_kernel()
    await kernel_pool.release_kernel(kernel_id)
    assert kernel_pool.kernels[kernel_id].state == KernelState.HEALTHY

@pytest.mark.asyncio
async def test_release_kernel_with_failure_marks_as_failed(kernel_pool: KernelPool):
    """Tests that releasing a failed kernel marks it as FAILED."""
    await kernel_pool.initialize()
    kernel_id = await kernel_pool.get_available_kernel()
    await kernel_pool.release_kernel(kernel_id, failed=True)
    assert kernel_pool.kernels[kernel_id].state == KernelState.FAILED
