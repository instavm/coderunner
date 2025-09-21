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
import socket

import aiofiles
import websockets
import httpx
# Import Context for progress reporting
from mcp.server.fastmcp import FastMCP, Context
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

# Import kernel pool components
from core.kernel_pool import (
    kernel_pool, 
    KernelError, 
    NoKernelAvailableError, 
    KernelExecutionError, 
    KernelTimeoutError,
    MAX_RETRY_ATTEMPTS,
    RETRY_BACKOFF_BASE,
    WEBSOCKET_TIMEOUT,
    WEBSOCKET_PING_INTERVAL,
    WEBSOCKET_PING_TIMEOUT,
    create_jupyter_request,
    JUPYTER_WS_URL
)

# --- CONFIGURATION & SETUP ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize the MCP server with a descriptive name for the toolset
mcp = FastMCP("CodeRunner")

def resolve_with_system_dns(hostname):
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror as e:
        print(f"Error resolving {hostname}: {e}")
        return None

PLAYWRIGHT_WS_URL = f"ws://127.0.0.1:3000/"

# --- ENHANCED EXECUTION WITH RETRY LOGIC ---

async def execute_with_retry(command: str, ctx: Context, max_attempts: int = MAX_RETRY_ATTEMPTS) -> str:
    """Execute command with retry logic and exponential backoff"""
    
    for attempt in range(max_attempts):
        try:
            # Get kernel from pool
            kernel_id = await kernel_pool.get_available_kernel()
            if not kernel_id:
                raise NoKernelAvailableError("No kernels available in pool")

            try:
                # Execute on kernel
                result = await _execute_on_kernel(kernel_id, command, ctx)
                # Release kernel back to pool on success
                await kernel_pool.release_kernel(kernel_id, failed=False)
                return result
            except Exception as e:
                # Release kernel as failed
                await kernel_pool.release_kernel(kernel_id, failed=True)
                raise e

        except Exception as e:
            if attempt == max_attempts - 1:
                # Last attempt failed, raise the error
                raise e
            
            # Calculate backoff delay
            delay = RETRY_BACKOFF_BASE ** attempt
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)

async def _execute_on_kernel(kernel_id: str, command: str, ctx: Context) -> str:
    """Execute code on a specific kernel with progress reporting"""
    jupyter_ws_url = f"{JUPYTER_WS_URL}/api/kernels/{kernel_id}/channels"
    final_output_lines = []
    execution_complete = False
    sent_msg_id = None

    try:
        async with websockets.connect(
            jupyter_ws_url,
            ping_interval=WEBSOCKET_PING_INTERVAL,
            ping_timeout=WEBSOCKET_PING_TIMEOUT,
            close_timeout=10
        ) as jupyter_ws:
            # Send execution request
            sent_msg_id, jupyter_request_json = create_jupyter_request(command)
            await jupyter_ws.send(jupyter_request_json)
            logger.info(f"Sent execute_request to kernel {kernel_id} (msg_id: {sent_msg_id})")

            # Report initial progress
            await ctx.report_progress(f"Executing on kernel {kernel_id}...")

            start_time = time.time()
            
            while not execution_complete and (time.time() - start_time) < WEBSOCKET_TIMEOUT:
                try:
                    message_str = await asyncio.wait_for(jupyter_ws.recv(), timeout=30.0)
                except asyncio.TimeoutError:
                    continue

                try:
                    message_data = json.loads(message_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse WebSocket message: {e}")
                    continue

                parent_msg_id = message_data.get("parent_header", {}).get("msg_id")
                if parent_msg_id != sent_msg_id:
                    continue

                msg_type = message_data.get("header", {}).get("msg_type")
                content = message_data.get("content", {})

                if msg_type == "stream":
                    stream_text = content.get("text", "")
                    final_output_lines.append(stream_text)
                    # Report intermediate output
                    await ctx.report_progress(stream_text)

                elif msg_type in ["execute_result", "display_data"]:
                    data = content.get("data", {})
                    
                    # Handle different output types
                    if "text/plain" in data:
                        result_text = data["text/plain"]
                        final_output_lines.append(result_text)
                        await ctx.report_progress(result_text)
                    
                    # Handle images
                    if "image/png" in data:
                        image_data = data["image/png"]
                        try:
                            image_bytes = base64.b64decode(image_data)
                            image_path = f"/app/uploads/output_{int(time.time())}.png"
                            
                            async with aiofiles.open(image_path, 'wb') as f:
                                await f.write(image_bytes)
                            
                            final_output_lines.append(f"[Image saved to {image_path}]")
                            await ctx.report_progress(f"Image saved to {image_path}")
                        except (binascii.Error, OSError) as e:
                            logger.error(f"Failed to save image: {e}")

                elif msg_type == "error":
                    ename = content.get("ename", "Error")
                    evalue = content.get("evalue", "")
                    traceback = content.get("traceback", [])
                    
                    error_msg = f"{ename}: {evalue}"
                    if traceback:
                        error_msg += "\n" + "\n".join(traceback)
                    
                    final_output_lines.append(error_msg)
                    await ctx.report_progress(f"Error: {error_msg}")

                elif msg_type == "status":
                    execution_state = content.get("execution_state")
                    if execution_state == "idle":
                        execution_complete = True
                        await ctx.report_progress("Execution completed")
                    elif execution_state == "busy":
                        await ctx.report_progress("Kernel is processing...")

            if not execution_complete:
                elapsed = time.time() - start_time
                raise KernelTimeoutError(f"Execution timed out after {elapsed:.0f} seconds")

            # The operation is complete, return the final result.
            final_output = "".join(final_output_lines).strip()
            return final_output if final_output else "Code executed successfully (no output)"

    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"WebSocket connection closed unexpectedly: {e}")
        raise KernelExecutionError(f"Connection to kernel lost: {e}")
    except asyncio.TimeoutError:
        raise KernelTimeoutError(f"Kernel {kernel_id} timed out during execution")
    except Exception as e:
        logger.error(f"Unexpected error during execution on kernel {kernel_id}: {e}")
        raise KernelExecutionError(f"Execution failed: {e}")

# --- MCP TOOLS ---

@mcp.tool()
async def execute_python_code(command: str, ctx: Context) -> str:
    """
    Execute Python code in a shared Jupyter notebook environment.
    
    This tool allows you to run Python code with full access to libraries like pandas, numpy, matplotlib, etc.
    All variables and imports persist between executions, allowing you to build complex analyses step by step.
    
    Args:
        command: The Python code to execute
    """
    try:
        # Initialize kernel pool if not already done
        if not kernel_pool._initialized:
            await kernel_pool.initialize()
        
        return await execute_with_retry(command, ctx)
    except Exception as e:
        logger.error(f"Failed to execute Python code: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
async def navigate_and_get_all_visible_text(url: str) -> str:
    """
    Navigate to a URL and extract all visible text from the page.
    
    This tool uses Playwright to load a web page and extract all visible text content,
    making it useful for reading articles, documentation, or any web content.
    
    Args:
        url: The URL to navigate to and extract text from
    """
    try:
        async with async_playwright() as p:
            # Connect to the existing Playwright server
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