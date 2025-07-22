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

import aiofiles
import websockets
from mcp.server.fastmcp import FastMCP
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


# Jupyter connection settings
JUPYTER_WS_URL = "ws://127.0.0.1:8888"

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

PLAYWRIGHT_WS_URL =f"ws://{resolve_with_system_dns('play20.local')}:3000/"



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


# --- MCP TOOLS ---

@mcp.tool()
async def execute_python_code(command: str) -> str:
    """
    Executes a string of Python code in a persistent Jupyter kernel and returns the output.
    This is suitable for calculations, data analysis, and interacting with previously defined variables.

    Args:
        command: The Python code to execute as a single string.
    """
    # 1. Get Kernel ID
    if not os.path.exists(KERNEL_ID_FILE_PATH):
        logger.error(f"Kernel ID file not found at: {KERNEL_ID_FILE_PATH}")
        return "Error: Kernel is not running. The kernel ID file was not found."

    with open(KERNEL_ID_FILE_PATH, 'r') as file:
        kernel_id = file.read().strip()

    if not kernel_id:
        return "Error: Kernel ID file is empty. Cannot connect to the kernel."

    # 2. Connect and Execute via WebSocket
    jupyter_ws_url = f"{JUPYTER_WS_URL}/api/kernels/{kernel_id}/channels"
    output_lines = []
    sent_msg_id = None

    try:
        async with websockets.connect(jupyter_ws_url) as jupyter_ws:
            sent_msg_id, jupyter_request_json = create_jupyter_request(command)
            await jupyter_ws.send(jupyter_request_json)
            logger.info(f"Sent execute_request (msg_id: {sent_msg_id})")

            execution_complete = False
            loop_timeout = 3600.0  # Total time to wait for a result
            start_time = time.time()

            while not execution_complete and (time.time() - start_time) < loop_timeout:
                try:
                    # Wait for a message with a short timeout to keep the loop responsive
                    message_str = await asyncio.wait_for(jupyter_ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                message_data = json.loads(message_str)
                parent_msg_id = message_data.get("parent_header", {}).get("msg_id")

                # Ignore messages not related to our request
                if parent_msg_id != sent_msg_id:
                    continue

                msg_type = message_data.get("header", {}).get("msg_type")
                content = message_data.get("content", {})

                if msg_type == "stream":
                    output_lines.append(content.get("text", ""))
                elif msg_type == "execute_result" or msg_type == "display_data":
                    output_lines.append(content.get("data", {}).get("text/plain", ""))
                elif msg_type == "error":
                    error_traceback = "\n".join(content.get("traceback", []))
                    logger.error(f"Execution error for msg_id {sent_msg_id}:\n{error_traceback}")
                    return f"Execution Error:\n{error_traceback}"
                elif msg_type == "status" and content.get("execution_state") == "idle":
                    # The kernel is idle, meaning our execution is finished.
                    execution_complete = True

            if not execution_complete:
                 logger.error(f"Execution timed out for msg_id: {sent_msg_id}")
                 return f"Error: Execution timed out after {loop_timeout} seconds."

            return "".join(output_lines) if output_lines else "[Execution successful with no output]"

    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"WebSocket connection failed: {e}")
        return f"Error: Could not connect to the Jupyter kernel. It may be offline. Details: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during execution: {e}", exc_info=True)
        return f"Error: An internal server error occurred: {str(e)}"

@mcp.tool()
async def open_page(url: str) -> str:
    """
    Opens a webpage using Playwright.

    Args:
        url: The URL of the webpage to open.
    """

    try:
        async with async_playwright() as p:
            browser = await p.chromium.connect(PLAYWRIGHT_WS_URL)
            page = await browser.new_page()
            await page.goto(url)
            await browser.close()

            return f"Page opened successfully: {url}"

    except Exception as e:
        logger.error(f"Failed to open page: {e}")
        return f"Error: Failed to open page: {str(e)}"


@mcp.tool()
async def click_element_on_page(selector: str) -> str:
    """
    Clicks an element specified by the selector using Playwright.

    Args:
        selector: The CSS selector of the element to click.
    """

    try:
        async with async_playwright() as p:
            browser = await p.chromium.connect(PLAYWRIGHT_WS_URL)
            page = await browser.new_page()

            # Assuming that the page is already navigated to the desired URL
            locator = page.locator(selector)
            await locator.click()
            await browser.close()

            return f"Clicked element with selector: {selector}"

    except Exception as e:
        logger.error(f"Failed to click element: {e}")
        return f"Error: Failed to click element: {str(e)}"


@mcp.tool()
async def get_visible_text_on_page(selector: str) -> str:
    """
    Retrieves visible text from a specified element using a selector with Playwright.

    Args:
        selector: The CSS selector of the element whose text is to be retrieved.
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.connect(PLAYWRIGHT_WS_URL)
            page = await browser.new_page()

            # Assuming the page is already navigated to the desired URL
            element_text = await page.text_content(selector)
            await browser.close()

            return element_text or f"No visible text found for selector: {selector}"

    except Exception as e:
        logger.error(f"Failed to retrieve visible text: {e}")
        return f"Error: Failed to retrieve visible text: {str(e)}"

@mcp.tool()
async def navigate_and_get_all_visible_text(url: str) -> str:
    """
    Retrieves all visible text from the entire webpage using Playwright.

    Args:
        url: The URL of the webpage from which to retrieve text.
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.connect("ws://play20.local:3000/")
            page = await browser.new_page()
            await page.goto(url)

            # Get the full HTML content
            html_content = await page.content()

            # Use BeautifulSoup to parse the HTML and extract visible text
            soup = BeautifulSoup(html_content, 'html.parser')
            visible_text = soup.get_text(separator="\n", strip=True)

            await browser.close()

            return visible_text

    except Exception as e:
        logger.error(f"Failed to retrieve all visible text: {e}")
        return f"Error: Failed to retrieve all visible text: {str(e)}"


app = mcp.sse_app()
