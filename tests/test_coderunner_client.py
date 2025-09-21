"""Tests for CodeRunner client functionality"""

import pytest
from unittest.mock import patch, MagicMock
import requests
from __init__ import CodeRunner
from core.exceptions import CodeRunnerError, SessionError, ExecutionError


class TestCodeRunnerClient:
    """Test CodeRunner client functionality"""
    
    def test_init_no_autostart(self):
        """Test CodeRunner initialization without auto-start"""
        runner = CodeRunner(auto_start=False)
        
        assert runner.base_url == "http://localhost:8223"
        assert runner.session_id is None
        assert isinstance(runner._session, requests.Session)
        
    def test_init_custom_base_url(self):
        """Test CodeRunner initialization with custom base URL"""
        custom_url = "http://test:9999"
        runner = CodeRunner(auto_start=False, base_url=custom_url)
        
        assert runner.base_url == custom_url
        
    @patch('__init__.ContainerManager.ensure_running')
    def test_init_autostart_success(self, mock_ensure):
        """Test CodeRunner initialization with successful auto-start"""
        mock_ensure.return_value = True
        
        runner = CodeRunner(auto_start=True)
        
        mock_ensure.assert_called_once()
        assert runner.base_url == "http://localhost:8223"
        
    @patch('__init__.ContainerManager.ensure_running')
    def test_init_autostart_failure(self, mock_ensure):
        """Test CodeRunner initialization with failed auto-start"""
        mock_ensure.side_effect = Exception("Docker not found")
        
        with pytest.raises(CodeRunnerError) as exc_info:
            CodeRunner(auto_start=True)
            
        assert "Failed to start CodeRunner" in str(exc_info.value)
        
    @patch('requests.Session.post')
    def test_create_session_success(self, mock_post):
        """Test successful session creation"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"session_id": "test-session-123"}
        mock_post.return_value = mock_response
        
        runner = CodeRunner(auto_start=False)
        session_id = runner._create_session()
        
        assert session_id == "test-session-123"
        mock_post.assert_called_once_with("http://localhost:8223/sessions")
        
    @patch('requests.Session.post')
    def test_create_session_failure(self, mock_post):
        """Test failed session creation"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"detail": "Internal server error"}
        mock_response.text = "Internal server error"
        mock_post.return_value = mock_response
        
        runner = CodeRunner(auto_start=False)
        
        with pytest.raises(SessionError) as exc_info:
            runner._create_session()
            
        assert "Failed to create session" in str(exc_info.value)
        
    @patch('requests.Session.post')
    def test_execute_success(self, mock_post):
        """Test successful code execution"""
        # Mock session creation
        session_response = MagicMock()
        session_response.status_code = 200
        session_response.json.return_value = {"session_id": "test-session"}
        
        # Mock execution
        exec_response = MagicMock()
        exec_response.status_code = 200
        exec_response.json.return_value = {
            "stdout": "Hello World!",
            "stderr": "",
            "execution_time": 0.1,
            "session_id": "test-session"
        }
        
        mock_post.side_effect = [session_response, exec_response]
        
        runner = CodeRunner(auto_start=False)
        result = runner.execute("print('Hello World!')")
        
        assert result["stdout"] == "Hello World!"
        assert result["stderr"] == ""
        assert result["execution_time"] == 0.1
        assert runner.session_id == "test-session"
        
        # Check execute call
        exec_call = mock_post.call_args_list[1]
        exec_data = exec_call[1]["json"]
        assert exec_data["command"] == "print('Hello World!')"
        assert exec_data["language"] == "python"
        assert exec_data["timeout"] == 30
        
    @patch('requests.Session.post')
    def test_execute_with_session(self, mock_post):
        """Test code execution with existing session"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "stdout": "Result",
            "stderr": "",
            "execution_time": 0.05,
            "session_id": "existing-session"
        }
        mock_post.return_value = mock_response
        
        runner = CodeRunner(auto_start=False)
        runner.session_id = "existing-session"  # Pre-existing session
        
        result = runner.execute("x = 5")
        
        assert result["stdout"] == "Result"
        # Should only make one call (no session creation)
        mock_post.assert_called_once()
        
    @patch('requests.Session.post')
    def test_execute_failure(self, mock_post):
        """Test failed code execution"""
        # Mock session creation success
        session_response = MagicMock()
        session_response.status_code = 200
        session_response.json.return_value = {"session_id": "test-session"}
        
        # Mock execution failure
        exec_response = MagicMock()
        exec_response.status_code = 400
        exec_response.json.return_value = {"detail": "Syntax error"}
        mock_post.side_effect = [session_response, exec_response]
        
        runner = CodeRunner(auto_start=False)
        
        with pytest.raises(ExecutionError) as exc_info:
            runner.execute("invalid syntax!")
            
        assert "Execution failed" in str(exc_info.value)
        
    @patch('requests.Session.post')
    def test_execute_timeout(self, mock_post):
        """Test execution timeout"""
        # Mock session creation
        session_response = MagicMock()
        session_response.status_code = 200
        session_response.json.return_value = {"session_id": "test-session"}
        
        # Mock execution timeout
        mock_post.side_effect = [
            session_response,
            requests.exceptions.Timeout("Request timed out")
        ]
        
        runner = CodeRunner(auto_start=False)
        
        with pytest.raises(ExecutionError) as exc_info:
            runner.execute("time.sleep(100)", timeout=5)
            
        assert "timed out" in str(exc_info.value)
        
    @patch('requests.Session.post')
    def test_execute_async_success(self, mock_post):
        """Test successful async execution"""
        # Mock session creation
        session_response = MagicMock()
        session_response.status_code = 200
        session_response.json.return_value = {"session_id": "test-session"}
        
        # Mock async execution
        async_response = MagicMock()
        async_response.status_code = 200
        async_response.json.return_value = {"task_id": "task-123"}
        
        mock_post.side_effect = [session_response, async_response]
        
        runner = CodeRunner(auto_start=False)
        task_id = runner.execute_async("print('async test')")
        
        assert task_id == "task-123"
        
    @patch('requests.Session.get')
    def test_get_task_status_success(self, mock_get):
        """Test getting task status successfully"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "task-123",
            "status": "completed",
            "result": {"stdout": "async result"}
        }
        mock_get.return_value = mock_response
        
        runner = CodeRunner(auto_start=False)
        status = runner.get_task_status("task-123")
        
        assert status["status"] == "completed"
        assert status["result"]["stdout"] == "async result"
        
    @patch('requests.Session.get')
    def test_get_task_status_not_found(self, mock_get):
        """Test getting status of non-existent task"""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        runner = CodeRunner(auto_start=False)
        
        with pytest.raises(ExecutionError) as exc_info:
            runner.get_task_status("nonexistent-task")
            
        assert "Task not found" in str(exc_info.value)
        
    @patch('requests.Session.delete')
    def test_close_session_success(self, mock_delete):
        """Test successful session closure"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_delete.return_value = mock_response
        
        runner = CodeRunner(auto_start=False)
        runner.session_id = "test-session"
        
        runner.close_session()
        
        assert runner.session_id is None
        mock_delete.assert_called_once_with("http://localhost:8223/sessions/test-session")
        
    @patch('requests.Session.delete')
    def test_close_session_no_session(self, mock_delete):
        """Test closing session when no session exists"""
        runner = CodeRunner(auto_start=False)
        
        runner.close_session()
        
        # Should not make any calls
        mock_delete.assert_not_called()
        
    @patch('requests.Session.get')
    def test_is_session_active_true(self, mock_get):
        """Test checking active session"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        runner = CodeRunner(auto_start=False)
        runner.session_id = "test-session"
        
        assert runner.is_session_active() == True
        
    @patch('requests.Session.get')
    def test_is_session_active_false(self, mock_get):
        """Test checking inactive session"""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        runner = CodeRunner(auto_start=False)
        runner.session_id = "test-session"
        
        assert runner.is_session_active() == False
        
    def test_is_session_active_no_session(self):
        """Test checking session activity when no session exists"""
        runner = CodeRunner(auto_start=False)
        
        assert runner.is_session_active() == False
        
    def test_context_manager(self):
        """Test CodeRunner as context manager"""
        with patch('requests.Session.delete') as mock_delete:
            mock_delete.return_value = MagicMock(status_code=200)
            
            with CodeRunner(auto_start=False) as runner:
                runner.session_id = "test-session"
                assert runner.session_id == "test-session"
                
            # Session should be closed after exiting context
            mock_delete.assert_called_once()