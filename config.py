import os
import pathlib
from typing import Optional


class Config:
    """Configuration settings for CodeRunner"""
    
    def __init__(self, **kwargs):
        # Jupyter settings
        self.jupyter_port = int(os.getenv("CODERUNNER_JUPYTER_PORT", kwargs.get("jupyter_port", 8888)))
        self.jupyter_host = os.getenv("CODERUNNER_JUPYTER_HOST", kwargs.get("jupyter_host", "127.0.0.1"))
        self.jupyter_ws_url = os.getenv("CODERUNNER_JUPYTER_WS_URL", kwargs.get("jupyter_ws_url", f"ws://{self.jupyter_host}:{self.jupyter_port}"))
        
        # Directory settings
        default_shared_dir = "./uploads" if not os.path.exists("/app") else "/app/uploads"
        shared_dir_path = os.getenv("CODERUNNER_SHARED_DIR", kwargs.get("shared_dir", default_shared_dir))
        self.shared_dir = pathlib.Path(shared_dir_path)
        self.kernel_id_file = os.getenv("CODERUNNER_KERNEL_ID_FILE", kwargs.get("kernel_id_file", None))
        
        # Execution settings
        self.execution_timeout = float(os.getenv("CODERUNNER_EXECUTION_TIMEOUT", kwargs.get("execution_timeout", 300.0)))
        self.websocket_timeout = float(os.getenv("CODERUNNER_WEBSOCKET_TIMEOUT", kwargs.get("websocket_timeout", 1.0)))
        self.max_wait_jupyter = int(os.getenv("CODERUNNER_MAX_WAIT_JUPYTER", kwargs.get("max_wait_jupyter", 30)))
        
        # FastMCP settings
        self.fastmcp_host = os.getenv("CODERUNNER_FASTMCP_HOST", kwargs.get("fastmcp_host", "0.0.0.0"))
        self.fastmcp_port = int(os.getenv("CODERUNNER_FASTMCP_PORT", kwargs.get("fastmcp_port", 8222)))
        
        # Logging settings
        self.log_level = os.getenv("CODERUNNER_LOG_LEVEL", kwargs.get("log_level", "INFO"))
        self.log_format = os.getenv("CODERUNNER_LOG_FORMAT", kwargs.get("log_format", "%(asctime)s - %(levelname)s - %(message)s"))
        
        # Resource settings
        self.max_kernel_memory = os.getenv("CODERUNNER_MAX_KERNEL_MEMORY", kwargs.get("max_kernel_memory", None))
        self.max_kernel_cpu = os.getenv("CODERUNNER_MAX_KERNEL_CPU", kwargs.get("max_kernel_cpu", None))
        if self.max_kernel_cpu is not None:
            self.max_kernel_cpu = float(self.max_kernel_cpu)
        
        # Ensure shared directory exists
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        
        # Set kernel ID file path if not provided
        if self.kernel_id_file is None:
            self.kernel_id_file = str(self.shared_dir / "python_kernel_id.txt")
    
    @property
    def jupyter_ws_base_url(self) -> str:
        """Get the base WebSocket URL for Jupyter"""
        return f"ws://{self.jupyter_host}:{self.jupyter_port}"
    
    @property
    def jupyter_api_base_url(self) -> str:
        """Get the base API URL for Jupyter"""
        return f"http://{self.jupyter_host}:{self.jupyter_port}"


# Global config instance
config = Config()