import asyncio
import json
import logging
import os
import time
import uuid
from typing import Dict, Any, Optional, Tuple

import websockets
from websockets.exceptions import ConnectionClosed

from config import config

logger = logging.getLogger(__name__)


class JupyterExecutionError(Exception):
    """Exception raised when Jupyter code execution fails"""
    pass


class JupyterConnectionError(Exception):
    """Exception raised when connection to Jupyter fails"""
    pass


class JupyterClient:
    """Client for executing code in Jupyter kernels via WebSocket"""
    
    def __init__(self):
        self.kernel_id: Optional[str] = None
        self._load_kernel_id()
    
    def _load_kernel_id(self) -> None:
        """Load kernel ID from file"""
        if not os.path.exists(config.kernel_id_file):
            logger.error(f"Kernel ID file not found at: {config.kernel_id_file}")
            return
        
        try:
            with open(config.kernel_id_file, 'r') as file:
                self.kernel_id = file.read().strip()
                if not self.kernel_id:
                    logger.error("Kernel ID file is empty")
                    self.kernel_id = None
        except Exception as e:
            logger.error(f"Error reading kernel ID file: {e}")
            self.kernel_id = None
    
    def _create_execute_request(self, code: str) -> Tuple[str, str]:
        """
        Create a Jupyter execute_request message.
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
    
    async def execute_code(self, code: str) -> str:
        """
        Execute Python code in the Jupyter kernel and return the output.
        
        Args:
            code: The Python code to execute
            
        Returns:
            The execution output as a string
            
        Raises:
            JupyterConnectionError: If unable to connect to Jupyter
            JupyterExecutionError: If code execution fails
        """
        if not self.kernel_id:
            raise JupyterConnectionError("Kernel is not running. The kernel ID is not available.")
        
        jupyter_ws_url = f"{config.jupyter_ws_base_url}/api/kernels/{self.kernel_id}/channels"
        output_lines = []
        sent_msg_id = None
        
        try:
            async with websockets.connect(jupyter_ws_url) as websocket:
                # Send execution request
                sent_msg_id, jupyter_request_json = self._create_execute_request(code)
                await websocket.send(jupyter_request_json)
                logger.info(f"Sent execute_request (msg_id: {sent_msg_id})")
                
                # Process responses
                execution_complete = False
                start_time = time.time()
                
                while not execution_complete and (time.time() - start_time) < config.execution_timeout:
                    try:
                        message_str = await asyncio.wait_for(
                            websocket.recv(), 
                            timeout=config.websocket_timeout
                        )
                    except asyncio.TimeoutError:
                        continue
                    
                    try:
                        message_data = json.loads(message_str)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON received: {e}")
                        continue
                    
                    parent_msg_id = message_data.get("parent_header", {}).get("msg_id")
                    
                    # Ignore messages not related to our request
                    if parent_msg_id != sent_msg_id:
                        continue
                    
                    execution_complete = self._process_message(message_data, output_lines)
                
                if not execution_complete:
                    raise JupyterExecutionError(
                        f"Execution timed out after {config.execution_timeout} seconds"
                    )
                
                return "".join(output_lines) if output_lines else "[Execution successful with no output]"
                
        except ConnectionClosed as e:
            logger.error(f"WebSocket connection closed: {e}")
            raise JupyterConnectionError(f"Could not connect to Jupyter kernel: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during execution: {e}", exc_info=True)
            raise JupyterExecutionError(f"Internal error during execution: {str(e)}")
    
    def _process_message(self, message_data: Dict[str, Any], output_lines: list) -> bool:
        """
        Process a single message from Jupyter.
        
        Args:
            message_data: The parsed message data
            output_lines: List to append output to
            
        Returns:
            True if execution is complete, False otherwise
            
        Raises:
            JupyterExecutionError: If the message indicates an error
        """
        msg_type = message_data.get("header", {}).get("msg_type")
        content = message_data.get("content", {})
        
        if msg_type == "stream":
            output_lines.append(content.get("text", ""))
        elif msg_type in ["execute_result", "display_data"]:
            output_lines.append(content.get("data", {}).get("text/plain", ""))
        elif msg_type == "error":
            error_traceback = "\n".join(content.get("traceback", []))
            logger.error(f"Jupyter execution error: {error_traceback}")
            raise JupyterExecutionError(f"Execution Error:\n{error_traceback}")
        elif msg_type == "status" and content.get("execution_state") == "idle":
            # Execution is complete
            return True
        
        return False
    
    def reload_kernel_id(self) -> None:
        """Reload kernel ID from file (useful if kernel was restarted)"""
        self._load_kernel_id()
    
    def is_kernel_available(self) -> bool:
        """Check if kernel ID is available"""
        return self.kernel_id is not None