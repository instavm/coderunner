"""Tests for container manager functionality"""

import pytest
from unittest.mock import patch, MagicMock
from container_manager import ContainerManager


class TestContainerManager:
    """Test container management functionality"""
    
    def test_container_configuration(self):
        """Test container configuration constants"""
        assert ContainerManager.CONTAINER_NAME == "coderunner"
        assert ContainerManager.REST_PORT == 8223
        assert ContainerManager.MCP_PORT == 8222
        assert ContainerManager.JUPYTER_PORT == 8888
        assert ContainerManager.PLAYWRIGHT_PORT == 3000
        assert ContainerManager.DOCKER_IMAGE == "coderunner:latest"
        
    @patch('requests.get')
    def test_is_healthy_success(self, mock_get):
        """Test health check when container is healthy"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        assert ContainerManager.is_healthy() == True
        mock_get.assert_called_once_with(
            "http://localhost:8223/health", 
            timeout=3
        )
        
    @patch('requests.get')
    def test_is_healthy_failure(self, mock_get):
        """Test health check when container is unhealthy"""
        # Test non-200 status code
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response
        
        assert ContainerManager.is_healthy() == False
        
        # Test connection error
        mock_get.side_effect = Exception("Connection refused")
        assert ContainerManager.is_healthy() == False
        
    @patch('subprocess.run')
    def test_check_docker_success(self, mock_run):
        """Test Docker availability check when Docker is available"""
        # Mock successful docker --version and docker info
        mock_run.return_value = MagicMock()
        
        assert ContainerManager.check_docker() == True
        
        # Should check both docker --version and docker info
        assert mock_run.call_count == 2
        
    @patch('subprocess.run')
    def test_check_docker_failure(self, mock_run):
        """Test Docker availability check when Docker is not available"""
        # Mock docker command not found
        mock_run.side_effect = FileNotFoundError("docker command not found")
        
        assert ContainerManager.check_docker() == False
        
        # Test docker daemon not running
        mock_run.reset_mock()
        mock_run.side_effect = [
            MagicMock(),  # docker --version succeeds
            subprocess.CalledProcessError(1, "docker info")  # docker info fails
        ]
        
        assert ContainerManager.check_docker() == False
        
    @patch('platform.system')
    def test_show_docker_install_help(self, mock_system, capsys):
        """Test Docker installation help for different platforms"""
        # Test macOS
        mock_system.return_value = "Darwin"
        ContainerManager._show_docker_install_help()
        captured = capsys.readouterr()
        assert "Docker Desktop for Mac" in captured.out
        
        # Test Windows
        mock_system.return_value = "Windows"
        ContainerManager._show_docker_install_help()
        captured = capsys.readouterr()
        assert "Docker Desktop for Windows" in captured.out
        
        # Test Linux
        mock_system.return_value = "Linux"
        ContainerManager._show_docker_install_help()
        captured = capsys.readouterr()
        assert "Docker Engine" in captured.out
        
    @patch('subprocess.run')
    def test_pull_image_if_needed_exists(self, mock_run):
        """Test image pull when image already exists"""
        # Mock image exists locally
        mock_result = MagicMock()
        mock_result.stdout = "sha256:abc123"
        mock_run.return_value = mock_result
        
        ContainerManager._pull_image_if_needed()
        
        # Should only check for image, not pull
        mock_run.assert_called_once()
        
    @patch('subprocess.run')
    def test_pull_image_if_needed_missing(self, mock_run):
        """Test image pull when image is missing"""
        # Mock image doesn't exist, then pull succeeds
        mock_run.side_effect = [
            MagicMock(stdout=""),  # docker images returns empty
            MagicMock()  # docker pull succeeds
        ]
        
        ContainerManager._pull_image_if_needed()
        
        # Should check for image and then pull
        assert mock_run.call_count == 2
        
    @patch('subprocess.run')
    def test_start_container_new(self, mock_run):
        """Test starting new container"""
        # Mock no existing container
        mock_run.side_effect = [
            MagicMock(stdout=""),  # docker ps -a returns empty
            MagicMock()  # docker run succeeds
        ]
        
        ContainerManager.start_container()
        
        # Should check for existing container and then create new one
        assert mock_run.call_count == 2
        
        # Check docker run command
        run_call = mock_run.call_args_list[1]
        run_args = run_call[0][0]
        assert "docker" in run_args
        assert "run" in run_args
        assert "-d" in run_args
        assert "--name" in run_args
        assert "coderunner" in run_args
        
    @patch('subprocess.run')
    def test_start_container_existing_stopped(self, mock_run):
        """Test starting existing but stopped container"""
        # Mock existing but stopped container
        mock_run.side_effect = [
            MagicMock(stdout="Exited (0) 1 hour ago"),  # docker ps -a
            MagicMock()  # docker start
        ]
        
        ContainerManager.start_container()
        
        # Should start existing container
        assert mock_run.call_count == 2
        
        # Check docker start command
        start_call = mock_run.call_args_list[1]
        start_args = start_call[0][0]
        assert "docker" in start_args
        assert "start" in start_args
        assert "coderunner" in start_args
        
    @patch('subprocess.run')
    def test_stop_container_success(self, mock_run):
        """Test stopping container successfully"""
        mock_run.return_value = MagicMock()
        
        result = ContainerManager.stop_container()
        
        assert result == True
        mock_run.assert_called_once_with(
            ["docker", "stop", "coderunner"],
            check=True,
            capture_output=True
        )
        
    @patch('subprocess.run')
    def test_stop_container_failure(self, mock_run):
        """Test stopping container when container doesn't exist"""
        mock_run.side_effect = subprocess.CalledProcessError(1, "docker stop")
        
        result = ContainerManager.stop_container()
        
        assert result == False
        
    @patch('subprocess.run')
    def test_get_container_status_running(self, mock_run):
        """Test getting status of running container"""
        mock_run.return_value = MagicMock(
            stdout="STATUS\tPORTS\nUp 5 minutes\t8223->8223, 8222->8222"
        )
        
        with patch.object(ContainerManager, 'is_healthy', return_value=True):
            status = ContainerManager.get_container_status()
            
        assert status["exists"] == True
        assert status["running"] == True
        assert status["healthy"] == True
        assert "Up 5 minutes" in status["status"]
        
    @patch('subprocess.run')
    def test_get_container_status_not_exists(self, mock_run):
        """Test getting status when container doesn't exist"""
        mock_run.return_value = MagicMock(stdout="STATUS\tPORTS\n")
        
        status = ContainerManager.get_container_status()
        
        assert status["exists"] == False
        assert status["running"] == False
        assert status["healthy"] == False