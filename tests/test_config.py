import os
import pathlib
import tempfile
from unittest.mock import patch

import pytest

from config import Config


def test_config_defaults():
    """Test that config has expected default values"""
    config = Config()
    
    assert config.jupyter_ws_url == "ws://127.0.0.1:8888"
    assert config.jupyter_port == 8888
    assert config.jupyter_host == "127.0.0.1"
    assert config.shared_dir == pathlib.Path("/app/uploads")
    assert config.execution_timeout == 300.0
    assert config.websocket_timeout == 1.0
    assert config.max_wait_jupyter == 30
    assert config.fastmcp_host == "0.0.0.0"
    assert config.fastmcp_port == 8222
    assert config.log_level == "INFO"


def test_config_from_env():
    """Test that config can be loaded from environment variables"""
    env_vars = {
        "CODERUNNER_JUPYTER_PORT": "9999",
        "CODERUNNER_JUPYTER_HOST": "192.168.1.1",
        "CODERUNNER_EXECUTION_TIMEOUT": "600.0",
        "CODERUNNER_FASTMCP_PORT": "8333",
        "CODERUNNER_LOG_LEVEL": "DEBUG"
    }
    
    with patch.dict(os.environ, env_vars):
        config = Config()
        
        assert config.jupyter_port == 9999
        assert config.jupyter_host == "192.168.1.1"
        assert config.execution_timeout == 600.0
        assert config.fastmcp_port == 8333
        assert config.log_level == "DEBUG"


def test_config_shared_dir_creation():
    """Test that shared directory is created if it doesn't exist"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        shared_dir = pathlib.Path(tmp_dir) / "test_uploads"
        
        # Directory shouldn't exist initially
        assert not shared_dir.exists()
        
        config = Config(shared_dir=shared_dir)
        
        # Directory should be created during initialization
        assert shared_dir.exists()
        assert shared_dir.is_dir()


def test_config_jupyter_urls():
    """Test that Jupyter URL properties work correctly"""
    config = Config(jupyter_host="localhost", jupyter_port=8888)
    
    assert config.jupyter_ws_base_url == "ws://localhost:8888"
    assert config.jupyter_api_base_url == "http://localhost:8888"


def test_config_kernel_id_file_path():
    """Test that kernel ID file path is set correctly"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        shared_dir = pathlib.Path(tmp_dir) / "uploads"
        config = Config(shared_dir=shared_dir)
        
        expected_path = str(shared_dir / "python_kernel_id.txt")
        assert config.kernel_id_file == expected_path