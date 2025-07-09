# --- IMPORTS ---
import logging

from mcp.server.fastmcp import FastMCP

from config import config
from jupyter_client import JupyterClient, JupyterConnectionError, JupyterExecutionError

# --- CONFIGURATION & SETUP ---
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format=config.log_format
)
logger = logging.getLogger(__name__)

# Initialize the MCP server with a descriptive name for the toolset
mcp = FastMCP("CodeRunner")

# Initialize Jupyter client
jupyter_client = JupyterClient()




# --- MCP TOOLS ---

@mcp.tool()
async def execute_python_code(command: str) -> str:
    """
    Executes a string of Python code in a persistent Jupyter kernel and returns the output.
    This is suitable for calculations, data analysis, and interacting with previously defined variables.

    Args:
        command: The Python code to execute as a single string.
    """
    try:
        result = await jupyter_client.execute_code(command, kernel_type="python")
        return result
    except JupyterConnectionError as e:
        logger.error(f"Jupyter connection error: {e}")
        return f"Error: {str(e)}"
    except JupyterExecutionError as e:
        logger.error(f"Jupyter execution error: {e}")
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return f"Error: An internal server error occurred: {str(e)}"


@mcp.tool()
async def execute_bash_code(command: str) -> str:
    """
    Executes a string of Bash shell commands in a persistent Jupyter bash kernel and returns the output.
    This is suitable for file operations, system commands, and shell scripting.

    Args:
        command: The Bash shell commands to execute as a single string.
    """
    try:
        result = await jupyter_client.execute_code(command, kernel_type="bash")
        return result
    except JupyterConnectionError as e:
        logger.error(f"Jupyter connection error: {e}")
        return f"Error: {str(e)}"
    except JupyterExecutionError as e:
        logger.error(f"Jupyter execution error: {e}")
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return f"Error: An internal server error occurred: {str(e)}"


@mcp.tool()
async def get_kernel_status() -> str:
    """
    Returns the status of available Jupyter kernels.
    
    Returns:
        A string describing which kernels are available and their status.
    """
    try:
        kernel_status = jupyter_client.get_available_kernels()
        status_lines = []
        
        for kernel_type, is_available in kernel_status.items():
            status = "✓ Available" if is_available else "✗ Not available"
            kernel_id = jupyter_client.kernels.get(kernel_type, "None")
            status_lines.append(f"{kernel_type.title()} kernel: {status} (ID: {kernel_id})")
        
        return "\n".join(status_lines)
    except Exception as e:
        logger.error(f"Error checking kernel status: {e}", exc_info=True)
        return f"Error: Could not check kernel status: {str(e)}"


app = mcp.sse_app()
