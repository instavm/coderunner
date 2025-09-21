"""
CodeRunner - Local code execution without API keys or cloud setup

Zero-configuration local code execution with seamless cloud migration.

Basic Usage (Isolated Fresh Container):
    from coderunner import CodeRunner
    
    runner = CodeRunner()  # Creates fresh isolated container
    result = runner.execute("print('Hello World!')")
    print(result['stdout'])  # "Hello World!"

Shared Container (for backward compatibility):
    runner = CodeRunner(isolated=False)  # Reuses shared container

Cloud Migration:
    from coderunner.cloud import InstaVM as CodeRunner
    
    runner = CodeRunner(api_key="your-key")  # Same interface!
    result = runner.execute("print('Hello Cloud!')")
"""

import requests
import logging
from typing import Dict, Optional, Any

from .container_manager import ContainerManager
from .core.exceptions import CodeRunnerError, SessionError, ExecutionError

__version__ = "1.0.0"

logger = logging.getLogger(__name__)


class CodeRunner:
    """Local code execution without API keys or cloud setup"""
    
    def __init__(self, auto_start: bool = True, base_url: str = None, isolated: bool = True):
        """
        Initialize CodeRunner client
        
        Args:
            auto_start: Automatically start container if not running
            base_url: Custom REST API base URL (for testing)
            isolated: Create isolated fresh container (True) or reuse shared (False)
        """
        self.isolated = isolated
        self.container_id: Optional[str] = None
        self._session = requests.Session()
        
        # Set reasonable timeouts
        self._session.timeout = 30
        
        if isolated and auto_start:
            # Create isolated fresh container
            try:
                self.container_id, ports = ContainerManager.create_isolated_container()
                self.base_url = f"http://localhost:{ports['mcp']}/api"
                logger.info(f"Created isolated container {self.container_id} on port {ports['mcp']}")
            except Exception as e:
                raise CodeRunnerError(f"Failed to create isolated container: {e}")
        else:
            # Use shared container (backward compatibility) - now unified on port 8222
            self.base_url = base_url or "http://localhost:8222/api"
            if auto_start:
                try:
                    ContainerManager.ensure_running()
                except Exception as e:
                    raise CodeRunnerError(f"Failed to start CodeRunner: {e}")
        
        self.session_id: Optional[str] = None
            
    def execute(self, code: str, language: str = "python", timeout: int = 30) -> Dict[str, Any]:
        """
        Execute code and return results (InstaVM compatible interface)
        
        Args:
            code: Code to execute
            language: Programming language ("python", "bash", "shell", "javascript")
            timeout: Execution timeout in seconds
            
        Returns:
            Dictionary with stdout, stderr, execution_time, etc.
            
        Raises:
            ExecutionError: If execution fails
            CodeRunnerError: If container communication fails
        """
        if not self.session_id:
            self.session_id = self._create_session()
            
        try:
            response = self._session.post(
                f"{self.base_url}/execute", 
                json={
                    "command": code,
                    "language": language,
                    "timeout": timeout,
                    "session_id": self.session_id
                }, 
                timeout=timeout + 10
            )
            
            if response.status_code != 200:
                error_detail = "Unknown error"
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", error_detail)
                except (ValueError, requests.exceptions.JSONDecodeError):
                    error_detail = response.text
                    
                raise ExecutionError(f"Execution failed: {error_detail}")
                
            return response.json()
            
        except requests.exceptions.Timeout:
            raise ExecutionError(f"Execution timed out after {timeout} seconds")
        except requests.exceptions.ConnectionError:
            raise CodeRunnerError(
                "Cannot connect to CodeRunner. Is the container running? "
                "Try CodeRunner().start_container()"
            )
        except ExecutionError:
            raise
        except Exception as e:
            raise CodeRunnerError(f"Unexpected error: {e}")
            
    def execute_async(self, code: str, language: str = "python", timeout: int = 30) -> str:
        """
        Execute code asynchronously, return task_id (InstaVM compatible)
        
        Args:
            code: Code to execute
            language: Programming language
            timeout: Execution timeout in seconds
            
        Returns:
            task_id string for checking execution status
            
        Raises:
            ExecutionError: If async execution setup fails
        """
        if not self.session_id:
            self.session_id = self._create_session()
            
        try:
            response = self._session.post(
                f"{self.base_url}/execute_async", 
                json={
                    "command": code,
                    "language": language,
                    "timeout": timeout,
                    "session_id": self.session_id
                }
            )
            
            if response.status_code != 200:
                error_detail = "Unknown error"
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", error_detail)
                except (ValueError, requests.exceptions.JSONDecodeError):
                    error_detail = response.text
                    
                raise ExecutionError(f"Async execution failed: {error_detail}")
                
            result = response.json()
            return result["task_id"]
            
        except requests.exceptions.ConnectionError:
            raise CodeRunnerError("Cannot connect to CodeRunner")
        except ExecutionError:
            raise
        except Exception as e:
            raise CodeRunnerError(f"Unexpected error: {e}")
            
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of async task (InstaVM compatible)
        
        Args:
            task_id: Task identifier from execute_async
            
        Returns:
            Task status dictionary
        """
        try:
            response = self._session.get(f"{self.base_url}/tasks/{task_id}")
            
            if response.status_code == 404:
                raise ExecutionError("Task not found")
            elif response.status_code != 200:
                raise ExecutionError(f"Failed to get task status: {response.text}")
                
            return response.json()
            
        except requests.exceptions.ConnectionError:
            raise CodeRunnerError("Cannot connect to CodeRunner")
        except ExecutionError:
            raise
        except Exception as e:
            raise CodeRunnerError(f"Unexpected error: {e}")
        
    def start_session(self) -> str:
        """
        Start a new execution session (InstaVM compatible)
        
        Returns:
            session_id string
            
        Raises:
            SessionError: If session creation fails
        """
        self.session_id = self._create_session()
        return self.session_id
        
    def close_session(self):
        """Close current session (InstaVM compatible)"""
        if self.session_id:
            try:
                self._session.delete(f"{self.base_url}/sessions/{self.session_id}")
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
            finally:
                self.session_id = None
                
    def cleanup(self):
        """Cleanup resources including isolated container if created"""
        self.close_session()
        
        if self.isolated and self.container_id:
            try:
                ContainerManager.remove_isolated_container(self.container_id)
                logger.info(f"Cleaned up isolated container {self.container_id}")
            except Exception as e:
                logger.warning(f"Error cleaning up container {self.container_id}: {e}")
            finally:
                self.container_id = None
                
    def is_session_active(self) -> bool:
        """
        Check if current session is active (InstaVM compatible)
        
        Returns:
            True if session is active, False otherwise
        """
        if not self.session_id:
            return False
            
        try:
            response = self._session.get(f"{self.base_url}/sessions/{self.session_id}")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
            
    def list_sessions(self) -> Dict[str, Any]:
        """
        List all active sessions
        
        Returns:
            Dictionary with session information
        """
        try:
            response = self._session.get(f"{self.base_url}/sessions")
            
            if response.status_code != 200:
                raise CodeRunnerError(f"Failed to list sessions: {response.text}")
                
            return response.json()
            
        except requests.exceptions.ConnectionError:
            raise CodeRunnerError("Cannot connect to CodeRunner")
        except Exception as e:
            raise CodeRunnerError(f"Unexpected error: {e}")
            
    def get_health(self) -> Dict[str, Any]:
        """
        Get CodeRunner health status
        
        Returns:
            Health status dictionary
        """
        try:
            response = self._session.get(f"{self.base_url}/health")
            
            if response.status_code != 200:
                raise CodeRunnerError(f"Health check failed: {response.text}")
                
            return response.json()
            
        except requests.exceptions.ConnectionError:
            raise CodeRunnerError("Cannot connect to CodeRunner")
        except Exception as e:
            raise CodeRunnerError(f"Unexpected error: {e}")
            
    def get_supported_languages(self) -> Dict[str, Any]:
        """
        Get list of supported programming languages
        
        Returns:
            Dictionary with supported languages
        """
        try:
            response = self._session.get(f"{self.base_url}/languages")
            
            if response.status_code != 200:
                raise CodeRunnerError(f"Failed to get languages: {response.text}")
                
            return response.json()
            
        except requests.exceptions.ConnectionError:
            raise CodeRunnerError("Cannot connect to CodeRunner")
        except Exception as e:
            raise CodeRunnerError(f"Unexpected error: {e}")
            
    def _create_session(self) -> str:
        """
        Internal method to create new session
        
        Returns:
            Session ID string
            
        Raises:
            SessionError: If session creation fails
        """
        try:
            response = self._session.post(f"{self.base_url}/sessions")
            
            if response.status_code != 200:
                error_detail = "Unknown error"
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", error_detail)
                except (ValueError, requests.exceptions.JSONDecodeError):
                    error_detail = response.text
                    
                raise SessionError(f"Failed to create session: {error_detail}")
                
            result = response.json()
            return result["session_id"]
            
        except requests.exceptions.ConnectionError:
            raise CodeRunnerError(
                "Cannot connect to CodeRunner. Is the container running? "
                "Try CodeRunner().start_container()"
            )
        except SessionError:
            raise
        except Exception as e:
            raise SessionError(f"Unexpected error creating session: {e}")
            
    # Container management convenience methods
    
    def start_container(self) -> bool:
        """Start CodeRunner container manually"""
        return ContainerManager.ensure_running()
        
    def stop_container(self) -> bool:
        """Stop CodeRunner container"""
        return ContainerManager.stop_container()
        
    def get_container_status(self) -> Dict[str, Any]:
        """Get container status information"""
        if self.isolated and self.container_id:
            # For isolated containers, get specific status
            containers = ContainerManager.list_isolated_containers()
            container_info = containers.get(self.container_id, {})
            return {
                "isolated": True,
                "container_id": self.container_id,
                "base_url": self.base_url,
                **container_info
            }
        else:
            # For shared container, get standard status
            status = ContainerManager.get_container_status()
            status["isolated"] = False
            return status
            
    def is_isolated(self) -> bool:
        """Check if this instance uses an isolated container"""
        return self.isolated
        
    def get_container_id(self) -> Optional[str]:
        """Get the container ID (for isolated containers)"""
        return self.container_id
        
    # Context manager support (InstaVM compatible)
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.isolated:
            self.cleanup()  # Cleanup isolated container
        else:
            self.close_session()  # Just close session for shared container


# Import cloud migration support
try:
    from . import cloud
    InstaVM = cloud.InstaVM
except ImportError:
    # Create placeholder for better error messages
    class InstaVM:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Cloud provider not available. Install with:\n"
                "  pip install coderunner[cloud]\n\n"
                "Or use local execution:\n"
                "  from coderunner import CodeRunner"
            )

__all__ = ["CodeRunner", "InstaVM"]