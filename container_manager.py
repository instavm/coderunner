"""Container management for CodeRunner auto-start functionality"""

import subprocess
import requests
import time
import os
import platform
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ContainerManager:
    """Manage CodeRunner Docker container lifecycle"""
    
    CONTAINER_NAME = "coderunner"
    REST_PORT = 8223
    MCP_PORT = 8222
    JUPYTER_PORT = 8888
    PLAYWRIGHT_PORT = 3000
    DOCKER_IMAGE = "coderunner:latest"
    
    @classmethod
    def ensure_running(cls) -> bool:
        """
        Ensure CodeRunner container is running - main entry point
        
        Returns:
            True if container is running and healthy
            
        Raises:
            RuntimeError: If Docker is not available or container fails to start
        """
        if cls.is_healthy():
            logger.info("CodeRunner container is already running")
            return True
            
        print("🚀 Starting CodeRunner container...")
        
        if not cls.check_docker():
            cls._show_docker_install_help()
            raise RuntimeError("Docker not found or not running")
            
        cls._pull_image_if_needed()
        cls.start_container()
        cls.wait_for_health()
        print("✅ CodeRunner ready!")
        return True
        
    @classmethod
    def is_healthy(cls) -> bool:
        """
        Check if CodeRunner REST API is responding
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = requests.get(
                f"http://localhost:{cls.REST_PORT}/health", 
                timeout=3
            )
            return response.status_code == 200
        except:
            return False
            
    @classmethod
    def check_docker(cls) -> bool:
        """
        Check if Docker is available and running
        
        Returns:
            True if Docker is available, False otherwise
        """
        try:
            # Check if Docker is installed
            subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                check=True
            )
            
            # Check if Docker daemon is running
            subprocess.run(
                ["docker", "info"], 
                capture_output=True, 
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
            
    @classmethod
    def _show_docker_install_help(cls):
        """Show Docker installation instructions based on platform"""
        system = platform.system()
        
        print("❌ Docker is required but not found or not running")
        print()
        
        if system == "Darwin":  # macOS
            print("📥 Install Docker Desktop for Mac:")
            print("   https://www.docker.com/products/docker-desktop")
        elif system == "Windows":
            print("📥 Install Docker Desktop for Windows:")
            print("   https://www.docker.com/products/docker-desktop")
        else:  # Linux
            print("📥 Install Docker Engine:")
            print("   https://docs.docker.com/get-docker/")
            
        print()
        print("After installation, make sure Docker is running before using CodeRunner.")
            
    @classmethod
    def _pull_image_if_needed(cls):
        """Pull Docker image if not available locally"""
        try:
            # Check if image exists locally
            result = subprocess.run(
                ["docker", "images", "-q", cls.DOCKER_IMAGE],
                capture_output=True, 
                text=True,
                check=True
            )
            
            if not result.stdout.strip():
                print("📥 Downloading CodeRunner container (first time setup)...")
                print("    This may take a few minutes...")
                
                subprocess.run(
                    ["docker", "pull", cls.DOCKER_IMAGE], 
                    check=True
                )
                print("✅ Container download complete")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull Docker image: {e}")
            raise RuntimeError(f"Failed to download CodeRunner container: {e}")
            
    @classmethod
    def start_container(cls):
        """Start CodeRunner container with proper configuration"""
        try:
            # Check if container exists
            result = subprocess.run(
                [
                    "docker", "ps", "-a", 
                    "--filter", f"name={cls.CONTAINER_NAME}", 
                    "--format", "{{.Status}}"
                ],
                capture_output=True, 
                text=True
            )
            
            if result.stdout.strip():
                # Container exists, check if it's running
                if "Up" in result.stdout:
                    logger.info("Container is already running")
                    return
                elif "Exited" in result.stdout:
                    # Container exists but stopped, start it
                    logger.info("Starting existing container")
                    subprocess.run(
                        ["docker", "start", cls.CONTAINER_NAME], 
                        check=True
                    )
                    return
            
            # Create and run new container
            logger.info("Creating new CodeRunner container")
            subprocess.run([
                "docker", "run", "-d",
                "--name", cls.CONTAINER_NAME,
                "-p", f"{cls.REST_PORT}:8223",
                "-p", f"{cls.MCP_PORT}:8222", 
                "-p", f"{cls.JUPYTER_PORT}:8888",
                "-p", f"{cls.PLAYWRIGHT_PORT}:3000",
                "--restart", "unless-stopped",
                cls.DOCKER_IMAGE
            ], check=True)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start container: {e}")
            raise RuntimeError(f"Failed to start CodeRunner container: {e}")
            
    @classmethod
    def wait_for_health(cls, timeout: int = 120):
        """
        Wait for container to be healthy
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Raises:
            TimeoutError: If container doesn't become healthy within timeout
        """
        start_time = time.time()
        last_log_time = 0
        
        print("⏳ Waiting for CodeRunner to initialize...")
        
        while time.time() - start_time < timeout:
            if cls.is_healthy():
                elapsed = time.time() - start_time
                print(f"✅ CodeRunner ready in {elapsed:.1f}s")
                return
                
            # Show progress every 10 seconds
            elapsed = time.time() - start_time
            if elapsed - last_log_time >= 10:
                print(f"   Still initializing... ({elapsed:.0f}s elapsed)")
                last_log_time = elapsed
                
            time.sleep(2)
            
        # Final check with container logs if timeout
        cls._show_container_logs()
        raise TimeoutError(
            f"CodeRunner container failed to start within {timeout} seconds. "
            "Check container logs above for details."
        )
        
    @classmethod
    def _show_container_logs(cls):
        """Show recent container logs for debugging"""
        try:
            print("\n📋 Recent container logs:")
            result = subprocess.run(
                ["docker", "logs", "--tail", "20", cls.CONTAINER_NAME],
                capture_output=True,
                text=True
            )
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
        except:
            print("   Could not retrieve container logs")
        
    @classmethod
    def stop_container(cls) -> bool:
        """
        Stop CodeRunner container
        
        Returns:
            True if stopped successfully, False if container not found
        """
        try:
            subprocess.run(
                ["docker", "stop", cls.CONTAINER_NAME], 
                check=True,
                capture_output=True
            )
            logger.info("CodeRunner container stopped")
            return True
        except subprocess.CalledProcessError:
            return False
        
    @classmethod
    def remove_container(cls) -> bool:
        """
        Remove CodeRunner container completely
        
        Returns:
            True if removed successfully, False if container not found
        """
        try:
            # Stop first if running
            subprocess.run(
                ["docker", "stop", cls.CONTAINER_NAME], 
                capture_output=True
            )
            
            # Remove container
            subprocess.run(
                ["docker", "rm", cls.CONTAINER_NAME], 
                check=True,
                capture_output=True
            )
            logger.info("CodeRunner container removed")
            return True
        except subprocess.CalledProcessError:
            return False
            
    @classmethod
    def get_container_status(cls) -> dict:
        """
        Get detailed container status information
        
        Returns:
            Dictionary with container status details
        """
        try:
            # Get container info
            result = subprocess.run(
                [
                    "docker", "ps", "-a", 
                    "--filter", f"name={cls.CONTAINER_NAME}",
                    "--format", "table {{.Status}}\t{{.Ports}}"
                ],
                capture_output=True,
                text=True
            )
            
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                return {
                    "exists": False,
                    "running": False,
                    "healthy": False,
                    "ports": []
                }
                
            status_line = lines[1]
            is_running = "Up" in status_line
            
            return {
                "exists": True,
                "running": is_running,
                "healthy": cls.is_healthy() if is_running else False,
                "status": status_line,
                "docker_available": cls.check_docker()
            }
            
        except Exception as e:
            logger.error(f"Error getting container status: {e}")
            return {
                "exists": False,
                "running": False,
                "healthy": False,
                "error": str(e),
                "docker_available": cls.check_docker()
            }