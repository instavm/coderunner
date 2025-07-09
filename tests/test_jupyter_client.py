import asyncio
import json
import os
import pathlib
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest
import websockets

from jupyter_client import JupyterClient, JupyterConnectionError, JupyterExecutionError
from config import Config


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        shared_dir = pathlib.Path(tmp_dir) / "uploads"
        shared_dir.mkdir()
        kernel_id_file = shared_dir / "python_kernel_id.txt"
        
        config = Config(
            shared_dir=shared_dir,
            kernel_id_file=str(kernel_id_file),
            execution_timeout=10.0,
            websocket_timeout=0.1
        )
        
        with patch('jupyter_client.config', config):
            yield config


@pytest.fixture
def jupyter_client_with_kernel(mock_config):
    """Create a JupyterClient instance with a mock kernel ID"""
    kernel_id = "test-kernel-123"
    
    # Write kernel ID to file
    with open(mock_config.kernel_id_file, 'w') as f:
        f.write(kernel_id)
    
    client = JupyterClient()
    return client


class TestJupyterClient:
    
    def test_init_no_kernel_file(self, mock_config):
        """Test initialization when kernel ID file doesn't exist"""
        client = JupyterClient()
        assert client.kernel_id is None
        assert not client.is_kernel_available()
    
    def test_init_empty_kernel_file(self, mock_config):
        """Test initialization when kernel ID file is empty"""
        # Create empty file
        with open(mock_config.kernel_id_file, 'w') as f:
            f.write("")
        
        client = JupyterClient()
        assert client.kernel_id is None
        assert not client.is_kernel_available()
    
    def test_init_with_kernel_file(self, jupyter_client_with_kernel):
        """Test initialization when kernel ID file exists and has content"""
        assert jupyter_client_with_kernel.kernel_id == "test-kernel-123"
        assert jupyter_client_with_kernel.is_kernel_available()
    
    def test_create_execute_request(self, jupyter_client_with_kernel):
        """Test creation of Jupyter execute request"""
        code = "print('hello world')"
        msg_id, request_json = jupyter_client_with_kernel._create_execute_request(code)
        
        assert msg_id is not None
        assert len(msg_id) > 0
        
        request = json.loads(request_json)
        assert request["header"]["msg_type"] == "execute_request"
        assert request["header"]["msg_id"] == msg_id
        assert request["content"]["code"] == code
        assert request["content"]["stop_on_error"] is True
    
    @pytest.mark.asyncio
    async def test_execute_code_no_kernel(self, mock_config):
        """Test execute_code when no kernel is available"""
        client = JupyterClient()
        
        with pytest.raises(JupyterConnectionError, match="Kernel is not running"):
            await client.execute_code("print('test')")
    
    @pytest.mark.asyncio
    async def test_execute_code_success(self, jupyter_client_with_kernel):
        """Test successful code execution"""
        code = "print('hello world')"
        expected_output = "hello world\n"
        
        # Mock WebSocket connection
        mock_websocket = AsyncMock()
        mock_websocket.send = AsyncMock()
        mock_websocket.recv = AsyncMock()
        
        # Mock the message sequence
        execute_request_msg = None
        def capture_request(msg):
            nonlocal execute_request_msg
            execute_request_msg = json.loads(msg)
        
        mock_websocket.send.side_effect = capture_request
        
        # Mock response messages
        async def mock_recv():
            if execute_request_msg is None:
                raise asyncio.TimeoutError()
            
            msg_id = execute_request_msg["header"]["msg_id"]
            
            # First return stream output
            stream_msg = {
                "header": {"msg_type": "stream"},
                "parent_header": {"msg_id": msg_id},
                "content": {"text": expected_output}
            }
            
            # Then return status idle
            status_msg = {
                "header": {"msg_type": "status"},
                "parent_header": {"msg_id": msg_id},
                "content": {"execution_state": "idle"}
            }
            
            # Return messages in sequence
            if not hasattr(mock_recv, 'call_count'):
                mock_recv.call_count = 0
            
            mock_recv.call_count += 1
            if mock_recv.call_count == 1:
                return json.dumps(stream_msg)
            else:
                return json.dumps(status_msg)
        
        mock_websocket.recv.side_effect = mock_recv
        
        # Mock websockets.connect
        with patch('websockets.connect') as mock_connect:
            mock_connect.return_value.__aenter__.return_value = mock_websocket
            
            result = await jupyter_client_with_kernel.execute_code(code)
            
            assert result == expected_output
            mock_websocket.send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_code_error(self, jupyter_client_with_kernel):
        """Test code execution with error"""
        code = "raise ValueError('test error')"
        
        # Mock WebSocket connection
        mock_websocket = AsyncMock()
        mock_websocket.send = AsyncMock()
        
        # Mock error response
        async def mock_recv():
            msg_id = "test-msg-id"
            error_msg = {
                "header": {"msg_type": "error"},
                "parent_header": {"msg_id": msg_id},
                "content": {"traceback": ["ValueError: test error"]}
            }
            return json.dumps(error_msg)
        
        mock_websocket.recv.side_effect = mock_recv
        
        # Mock the request creation to return predictable msg_id
        with patch.object(jupyter_client_with_kernel, '_create_execute_request') as mock_create:
            mock_create.return_value = ("test-msg-id", '{"test": "request"}')
            
            with patch('websockets.connect') as mock_connect:
                mock_connect.return_value.__aenter__.return_value = mock_websocket
                
                with pytest.raises(JupyterExecutionError, match="ValueError: test error"):
                    await jupyter_client_with_kernel.execute_code(code)
    
    @pytest.mark.asyncio
    async def test_execute_code_connection_error(self, jupyter_client_with_kernel):
        """Test code execution with connection error"""
        code = "print('test')"
        
        # Mock connection failure
        with patch('websockets.connect') as mock_connect:
            mock_connect.side_effect = websockets.exceptions.ConnectionClosed(None, None)
            
            with pytest.raises(JupyterConnectionError, match="Could not connect to Jupyter kernel"):
                await jupyter_client_with_kernel.execute_code(code)
    
    def test_reload_kernel_id(self, mock_config):
        """Test reloading kernel ID from file"""
        client = JupyterClient()
        assert client.kernel_id is None
        
        # Write kernel ID to file
        with open(mock_config.kernel_id_file, 'w') as f:
            f.write("new-kernel-456")
        
        client.reload_kernel_id()
        assert client.kernel_id == "new-kernel-456"
        assert client.is_kernel_available()