"""Tests for isolated container functionality"""

import pytest
from unittest.mock import patch, MagicMock
from __init__ import CodeRunner
from container_manager import ContainerManager
from core.exceptions import CodeRunnerError


class TestIsolatedContainers:
    """Test isolated container creation and management"""
    
    @patch.object(ContainerManager, 'create_isolated_container')
    def test_isolated_container_creation(self, mock_create):
        """Test creating isolated container"""
        # Mock isolated container creation
        mock_create.return_value = ("container123", {
            "rest": 8323,
            "mcp": 8322,
            "jupyter": 8988,
            "playwright": 3100
        })
        
        runner = CodeRunner(auto_start=True, isolated=True)
        
        assert runner.isolated == True
        assert runner.container_id == "container123"
        assert runner.base_url == "http://localhost:8323"
        mock_create.assert_called_once()
        
    @patch.object(ContainerManager, 'ensure_running')
    def test_shared_container_backward_compatibility(self, mock_ensure):
        """Test shared container mode for backward compatibility"""
        mock_ensure.return_value = True
        
        runner = CodeRunner(auto_start=True, isolated=False)
        
        assert runner.isolated == False
        assert runner.container_id is None
        assert runner.base_url == "http://localhost:8223"
        mock_ensure.assert_called_once()
        
    @patch.object(ContainerManager, 'create_isolated_container')
    def test_isolated_container_creation_failure(self, mock_create):
        """Test handling of isolated container creation failure"""
        mock_create.side_effect = RuntimeError("Docker not available")
        
        with pytest.raises(CodeRunnerError) as exc_info:
            CodeRunner(auto_start=True, isolated=True)
            
        assert "Failed to create isolated container" in str(exc_info.value)
        
    def test_no_auto_start(self):
        """Test creating CodeRunner without auto-start"""
        runner = CodeRunner(auto_start=False, isolated=True)
        
        assert runner.isolated == True
        assert runner.container_id is None  # No container created
        assert runner.base_url == "http://localhost:8223"  # Default URL
        
    @patch.object(ContainerManager, 'remove_isolated_container')
    def test_cleanup_isolated_container(self, mock_remove):
        """Test cleanup of isolated container"""
        mock_remove.return_value = True
        
        runner = CodeRunner(auto_start=False, isolated=True)
        runner.container_id = "container123"
        
        runner.cleanup()
        
        mock_remove.assert_called_once_with("container123")
        assert runner.container_id is None
        
    @patch.object(ContainerManager, 'remove_isolated_container')
    def test_context_manager_cleanup(self, mock_remove):
        """Test context manager cleanup for isolated container"""
        mock_remove.return_value = True
        
        runner = CodeRunner(auto_start=False, isolated=True)
        runner.container_id = "container123"
        
        with runner:
            pass  # Container should be cleaned up on exit
            
        mock_remove.assert_called_once_with("container123")
        
    @patch.object(ContainerManager, 'list_isolated_containers')
    def test_get_container_status_isolated(self, mock_list):
        """Test getting status of isolated container"""
        mock_list.return_value = {
            "container123": {
                "id": "container123",
                "name": "coderunner-abc123",
                "status": "Up 5 minutes",
                "ports": "8323->8223, 8322->8222"
            }
        }
        
        runner = CodeRunner(auto_start=False, isolated=True)
        runner.container_id = "container123"
        runner.base_url = "http://localhost:8323"
        
        status = runner.get_container_status()
        
        assert status["isolated"] == True
        assert status["container_id"] == "container123"
        assert status["base_url"] == "http://localhost:8323"
        assert status["name"] == "coderunner-abc123"
        
    @patch.object(ContainerManager, 'get_container_status')
    def test_get_container_status_shared(self, mock_status):
        """Test getting status of shared container"""
        mock_status.return_value = {
            "exists": True,
            "running": True,
            "healthy": True
        }
        
        runner = CodeRunner(auto_start=False, isolated=False)
        status = runner.get_container_status()
        
        assert status["isolated"] == False
        assert status["exists"] == True
        assert status["running"] == True
        
    def test_isolation_methods(self):
        """Test isolation checking methods"""
        # Isolated container
        isolated_runner = CodeRunner(auto_start=False, isolated=True)
        isolated_runner.container_id = "container123"
        
        assert isolated_runner.is_isolated() == True
        assert isolated_runner.get_container_id() == "container123"
        
        # Shared container
        shared_runner = CodeRunner(auto_start=False, isolated=False)
        
        assert shared_runner.is_isolated() == False
        assert shared_runner.get_container_id() is None
        
    @patch.object(ContainerManager, 'create_isolated_container')
    def test_multiple_isolated_instances(self, mock_create):
        """Test creating multiple isolated instances"""
        # Mock different containers for each instance
        mock_create.side_effect = [
            ("container1", {"rest": 8323, "mcp": 8322, "jupyter": 8988, "playwright": 3100}),
            ("container2", {"rest": 8423, "mcp": 8422, "jupyter": 9088, "playwright": 3200}),
        ]
        
        runner1 = CodeRunner(auto_start=True, isolated=True)
        runner2 = CodeRunner(auto_start=True, isolated=True)
        
        # Each instance should have different containers and ports
        assert runner1.container_id == "container1"
        assert runner1.base_url == "http://localhost:8323"
        
        assert runner2.container_id == "container2"
        assert runner2.base_url == "http://localhost:8423"
        
        # Both instances are isolated
        assert runner1.is_isolated() == True
        assert runner2.is_isolated() == True
        
        # Containers are different
        assert runner1.get_container_id() != runner2.get_container_id()