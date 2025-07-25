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

PLAYWRIGHT_WS_URL =f"ws://127.0.0.1:3000/"



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


app = mcp.sse_app()
