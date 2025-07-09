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
        result = await jupyter_client.execute_code(command)
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


app = mcp.sse_app()
