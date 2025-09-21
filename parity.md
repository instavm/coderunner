# CodeRunner-InstaVM Parity Implementation Plan

## Overview
This document contains the complete implementation plan for achieving parity between local CodeRunner and cloud InstaVM, enabling seamless migration with minimal code changes.

## Strategy Summary

### Primary Goal: Frictionless Local Usage
```python
# Zero configuration - no API keys, no setup
from coderunner import CodeRunner

runner = CodeRunner()  # Auto-starts container
result = runner.execute("print('Hello World')")
```

### Secondary Goal: 1-2 Line Cloud Migration  
```python
# Option A: Import alias (RECOMMENDED)
from coderunner.cloud import InstaVM as CodeRunner
runner = CodeRunner(api_key="your-key")  # Only difference

# Option B: Direct InstaVM import
from instavm import InstaVM
runner = InstaVM(api_key="your-key")
```

## Phase 1: Code Execution Implementation (Weeks 1-3)

### Week 1: Core Infrastructure

#### Day 1-2: REST API Server Addition

**File: `/coderunner/api/rest_server.py`**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid
from ..core.kernel_pool import kernel_pool

app = FastAPI(title="CodeRunner REST API", version="1.0.0")

class CommandRequest(BaseModel):
    command: str
    language: Optional[str] = "python"
    timeout: Optional[int] = 30
    session_id: Optional[str] = None

class ExecutionResponse(BaseModel):
    stdout: str
    stderr: str
    execution_time: float
    cpu_time: Optional[float] = None
    session_id: str

@app.post("/execute", response_model=ExecutionResponse)
async def execute_command(request: CommandRequest):
    """Execute command - InstaVM compatible interface"""
    try:
        # Get or create session
        session_id = request.session_id or await session_manager.create_session(request.language)
        
        # Process command based on language
        processed_command = LanguageProcessor.preprocess_command(request.command, request.language)
        
        # Execute using existing kernel pool
        result = await session_manager.execute_in_session(session_id, processed_command)
        
        return ExecutionResponse(
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            execution_time=result.get("execution_time", 0.0),
            cpu_time=result.get("cpu_time"),
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute_async")
async def execute_command_async(request: CommandRequest):
    """Async execution - InstaVM compatible"""
    task_id = str(uuid.uuid4())
    # Schedule async task using existing kernel pool
    await kernel_pool.execute_async(request.command, task_id)
    return {"task_id": task_id}

@app.post("/sessions")
async def create_session():
    """Create execution session"""
    session_id = await session_manager.create_session()
    return {"session_id": session_id}

@app.delete("/sessions/{session_id}")
async def close_session(session_id: str):
    """Close execution session"""
    await session_manager.close_session(session_id)
    return {"success": True}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}
```

**Docker Updates:**
```dockerfile
# Update Dockerfile
EXPOSE 8222 8223

# Update entrypoint.sh
uvicorn coderunner.api.rest_server:app --host 0.0.0.0 --port 8223 &
```

#### Day 3-4: Session Management System

**File: `/coderunner/core/session_manager.py`**
```python
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime, timedelta
import uuid
import asyncio

@dataclass
class Session:
    id: str
    kernel_id: str
    created_at: datetime
    last_used: datetime
    language: str = "python"

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self._cleanup_task = None
        
    async def initialize(self):
        """Start background cleanup task"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def create_session(self, language: str = "python") -> str:
        """Create new execution session with dedicated kernel"""
        session_id = str(uuid.uuid4())
        
        # Get dedicated kernel from existing pool
        kernel_id = await kernel_pool.get_dedicated_kernel()
        
        session = Session(
            id=session_id,
            kernel_id=kernel_id,
            created_at=datetime.now(),
            last_used=datetime.now(),
            language=language
        )
        
        self.sessions[session_id] = session
        return session_id
        
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        return self.sessions.get(session_id)
        
    async def execute_in_session(self, session_id: str, code: str) -> dict:
        """Execute code in specific session"""
        session = self.sessions.get(session_id)
        if not session:
            raise SessionError("Session not found")
        
        session.last_used = datetime.now()
        
        # Use existing kernel pool execution with specific kernel
        return await kernel_pool.execute_on_kernel(session.kernel_id, code)
        
    async def close_session(self, session_id: str):
        """Close session and release kernel"""
        session = self.sessions.get(session_id)
        if session:
            await kernel_pool.release_kernel(session.kernel_id)
            del self.sessions[session_id]
            
    async def _cleanup_loop(self):
        """Background task to cleanup old sessions"""
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes
            cutoff = datetime.now() - timedelta(hours=1)
            
            expired_sessions = [
                sid for sid, session in self.sessions.items()
                if session.last_used < cutoff
            ]
            
            for session_id in expired_sessions:
                await self.close_session(session_id)

# Global session manager instance
session_manager = SessionManager()

class SessionError(Exception):
    pass
```

**File: `/coderunner/core/language_processor.py`**
```python
class LanguageProcessor:
    """Handle multi-language execution like InstaVM"""
    
    @staticmethod
    def preprocess_command(command: str, language: str) -> str:
        """Preprocess command based on language"""
        if language.lower() in ("bash", "shell", "sh"):
            if not command.strip().startswith("%%bash"):
                return f"%%bash\n{command}"
        elif command.startswith("!"):
            # Convert ! commands to %%bash 
            return f"%%bash\n{command[1:]}"
        return command
        
    @staticmethod  
    def detect_language(command: str) -> str:
        """Auto-detect language from command"""
        if command.strip().startswith("%%bash"):
            return "bash"
        if command.strip().startswith("!"):
            return "bash"
        return "python"
```

#### Day 5: Kernel Pool Updates

**Update existing `/coderunner/core/kernel_pool.py`:**
```python
# Add session support to existing kernel pool
class KernelPool:
    # ... existing code ...
    
    async def get_dedicated_kernel(self) -> str:
        """Get a kernel dedicated to a session"""
        # Similar to get_available_kernel but marks as dedicated
        pass
        
    async def execute_on_kernel(self, kernel_id: str, code: str) -> dict:
        """Execute code on specific kernel"""
        # Direct execution on specified kernel
        pass
        
    async def execute_async(self, code: str, task_id: str):
        """Schedule async execution"""
        # Add to async task queue
        pass
```

### Week 2: CodeRunner Client Package

#### Day 1-2: Auto-Container Management

**File: `/coderunner/container_manager.py`**
```python
import subprocess
import requests
import time
import os
import platform
from pathlib import Path

class ContainerManager:
    CONTAINER_NAME = "coderunner"
    REST_PORT = 8223
    DOCKER_IMAGE = "coderunner:latest"
    
    @classmethod
    def ensure_running(cls) -> bool:
        """Ensure CodeRunner container is running - main entry point"""
        if cls.is_healthy():
            return True
            
        print("🚀 Starting CodeRunner container...")
        
        if not cls.check_docker():
            cls._show_docker_install_help()
            raise RuntimeError("Docker not found")
            
        cls._pull_image_if_needed()
        cls.start_container()
        cls.wait_for_health()
        print("✅ CodeRunner ready!")
        return True
        
    @classmethod
    def is_healthy(cls) -> bool:
        """Check if CodeRunner REST API is responding"""
        try:
            response = requests.get(f"http://localhost:{cls.REST_PORT}/health", timeout=3)
            return response.status_code == 200
        except:
            return False
            
    @classmethod
    def check_docker(cls) -> bool:
        """Check if Docker is available"""
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
            # Also check if Docker daemon is running
            subprocess.run(["docker", "info"], capture_output=True, check=True)
            return True
        except:
            return False
            
    @classmethod
    def _show_docker_install_help(cls):
        """Show Docker installation instructions"""
        system = platform.system()
        if system == "Darwin":  # macOS
            print("📥 Please install Docker Desktop for Mac:")
            print("   https://www.docker.com/products/docker-desktop")
        elif system == "Windows":
            print("📥 Please install Docker Desktop for Windows:")
            print("   https://www.docker.com/products/docker-desktop")
        else:  # Linux
            print("📥 Please install Docker:")
            print("   https://docs.docker.com/get-docker/")
            
    @classmethod
    def _pull_image_if_needed(cls):
        """Pull Docker image if not available locally"""
        result = subprocess.run(
            ["docker", "images", "-q", cls.DOCKER_IMAGE],
            capture_output=True, text=True
        )
        
        if not result.stdout.strip():
            print("📥 Downloading CodeRunner container (first time setup)...")
            subprocess.run(["docker", "pull", cls.DOCKER_IMAGE], check=True)
            
    @classmethod
    def start_container(cls):
        """Start CodeRunner container"""
        # Check if container exists but is stopped
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={cls.CONTAINER_NAME}", "--format", "{{.Status}}"],
            capture_output=True, text=True
        )
        
        if result.stdout.strip():
            if "Exited" in result.stdout:
                # Container exists but stopped, start it
                subprocess.run(["docker", "start", cls.CONTAINER_NAME], check=True)
            # else container is already running
        else:
            # Create and run new container
            subprocess.run([
                "docker", "run", "-d",
                "--name", cls.CONTAINER_NAME,
                "-p", f"{cls.REST_PORT}:8223",
                "-p", "8888:8888",
                "-p", "3000:3000",
                "--restart", "unless-stopped",  # Auto-restart
                cls.DOCKER_IMAGE
            ], check=True)
            
    @classmethod
    def wait_for_health(cls, timeout: int = 60):
        """Wait for container to be healthy"""
        start_time = time.time()
        last_status = ""
        
        while time.time() - start_time < timeout:
            if cls.is_healthy():
                return
                
            # Show progress
            elapsed = int(time.time() - start_time)
            if elapsed % 5 == 0 and elapsed != last_status:
                print(f"   Waiting for startup... ({elapsed}s)")
                last_status = elapsed
                
            time.sleep(2)
            
        raise TimeoutError("CodeRunner container failed to start within timeout")
        
    @classmethod
    def stop_container(cls):
        """Stop CodeRunner container"""
        subprocess.run(["docker", "stop", cls.CONTAINER_NAME], check=True)
        
    @classmethod
    def remove_container(cls):
        """Remove CodeRunner container completely"""
        subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)
```

#### Day 3-4: Main CodeRunner Client

**File: `/coderunner/__init__.py`**
```python
import requests
from typing import Dict, Optional, Any
from .container_manager import ContainerManager
from .exceptions import CodeRunnerError, SessionError, ExecutionError

class CodeRunner:
    """Local code execution without API keys or cloud setup"""
    
    def __init__(self, auto_start: bool = True):
        """
        Initialize CodeRunner client
        
        Args:
            auto_start: Automatically start container if not running
        """
        self.base_url = "http://localhost:8223"
        self.session_id: Optional[str] = None
        self._session = requests.Session()
        
        if auto_start:
            ContainerManager.ensure_running()
            
    def execute(self, code: str, language: str = "python", timeout: int = 30) -> Dict[str, Any]:
        """
        Execute code and return results (InstaVM compatible interface)
        
        Args:
            code: Code to execute
            language: Programming language ("python", "bash", "shell")
            timeout: Execution timeout in seconds
            
        Returns:
            Dictionary with stdout, stderr, execution_time, etc.
        """
        if not self.session_id:
            self.session_id = self._create_session()
            
        try:
            response = self._session.post(f"{self.base_url}/execute", json={
                "command": code,
                "language": language,
                "timeout": timeout,
                "session_id": self.session_id
            }, timeout=timeout + 10)
            
            if response.status_code != 200:
                raise ExecutionError(f"Execution failed: {response.text}")
                
            return response.json()
            
        except requests.exceptions.Timeout:
            raise ExecutionError(f"Execution timed out after {timeout} seconds")
        except requests.exceptions.ConnectionError:
            raise CodeRunnerError("Cannot connect to CodeRunner. Is the container running?")
            
    def execute_async(self, code: str, language: str = "python", timeout: int = 30) -> str:
        """
        Execute code asynchronously, return task_id (InstaVM compatible)
        
        Returns:
            task_id string for checking execution status
        """
        if not self.session_id:
            self.session_id = self._create_session()
            
        response = self._session.post(f"{self.base_url}/execute_async", json={
            "command": code,
            "language": language,
            "timeout": timeout,
            "session_id": self.session_id
        })
        
        if response.status_code != 200:
            raise ExecutionError(f"Async execution failed: {response.text}")
            
        result = response.json()
        return result["task_id"]
        
    def start_session(self) -> str:
        """
        Start a new execution session (InstaVM compatible)
        
        Returns:
            session_id string
        """
        self.session_id = self._create_session()
        return self.session_id
        
    def close_session(self):
        """Close current session (InstaVM compatible)"""
        if self.session_id:
            try:
                self._session.delete(f"{self.base_url}/sessions/{self.session_id}")
            except:
                pass  # Ignore errors during cleanup
            self.session_id = None
            
    def is_session_active(self) -> bool:
        """Check if current session is active (InstaVM compatible)"""
        if not self.session_id:
            return False
        try:
            response = self._session.get(f"{self.base_url}/sessions/{self.session_id}")
            return response.status_code == 200
        except:
            return False
            
    def _create_session(self) -> str:
        """Internal method to create new session"""
        try:
            response = self._session.post(f"{self.base_url}/sessions")
            if response.status_code != 200:
                raise SessionError(f"Failed to create session: {response.text}")
            result = response.json()
            return result["session_id"]
        except requests.exceptions.ConnectionError:
            raise CodeRunnerError("Cannot connect to CodeRunner. Is the container running?")
            
    # Context manager support (InstaVM compatible)
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_session()

# Import for cloud migration
from .cloud import InstaVM

__version__ = "1.0.0"
__all__ = ["CodeRunner", "InstaVM"]
```

**File: `/coderunner/exceptions.py`**
```python
"""Exception classes matching InstaVM interface"""

class CodeRunnerError(Exception):
    """Base exception for CodeRunner"""
    pass

class SessionError(CodeRunnerError):
    """Session-related errors"""
    pass

class ExecutionError(CodeRunnerError):
    """Code execution errors"""
    pass

class ContainerError(CodeRunnerError):
    """Container management errors"""
    pass
```

#### Day 5: Cloud Migration Support

**File: `/coderunner/cloud/__init__.py`**
```python
"""
Cloud execution via InstaVM - same interface as local CodeRunner

This module provides the cloud migration path for CodeRunner users.
Simply change your import to use cloud execution with the same interface.

Example migration:
    # Before (local)
    from coderunner import CodeRunner
    runner = CodeRunner()
    
    # After (cloud) - just add API key
    from coderunner.cloud import InstaVM as CodeRunner  
    runner = CodeRunner(api_key="your-key")
"""

try:
    from instavm import InstaVM
    
    # Add helpful docstring for migration
    InstaVM.__doc__ = """
    Cloud execution client (InstaVM) with CodeRunner-compatible interface.
    
    This is an alias to the InstaVM class for easy migration from local CodeRunner.
    All methods have the same interface as local CodeRunner.
    
    Usage:
        from coderunner.cloud import InstaVM as CodeRunner
        runner = CodeRunner(api_key="your-api-key")
        result = runner.execute("print('Hello Cloud!')")
    """
    
except ImportError:
    class InstaVM:
        """Placeholder InstaVM class when package not installed"""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "InstaVM package not found. Install with:\n"
                "  pip install instavm\n\n"
                "Or use local CodeRunner:\n"
                "  from coderunner import CodeRunner\n"
                "  runner = CodeRunner()"
            )

__all__ = ["InstaVM"]
```

### Week 3: Integration & Polish

#### Day 1-2: AI Framework Integrations

**File: `/coderunner/integrations/openai.py`**
```python
"""OpenAI integration for both local and cloud CodeRunner"""

from typing import List, Dict, Any, Union
import json

def get_tools() -> List[Dict[str, Any]]:
    """
    Get OpenAI function calling tools for CodeRunner (works with both local and cloud)
    
    Returns:
        List of tool definitions for OpenAI function calling
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "execute_python_code",
                "description": "Execute Python code securely. Use for data analysis, calculations, file operations, web requests, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string", 
                            "description": "Python code to execute. Can include imports, functions, classes, etc."
                        },
                        "language": {
                            "type": "string",
                            "enum": ["python", "bash", "shell"],
                            "default": "python",
                            "description": "Programming language (python for Python code, bash/shell for terminal commands)"
                        }
                    },
                    "required": ["code"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "execute_bash_command",
                "description": "Execute bash/shell commands. Use for system operations, file management, package installation, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Bash command to execute (e.g., 'ls -la', 'pip install requests', 'curl https://api.example.com')"
                        }
                    },
                    "required": ["command"]
                }
            }
        }
    ]

def execute_tool(runner: Union['CodeRunner', 'InstaVM'], tool_call) -> Dict[str, Any]:
    """
    Execute OpenAI tool call with CodeRunner (works with both local and cloud)
    
    Args:
        runner: CodeRunner or InstaVM instance
        tool_call: OpenAI tool call object
        
    Returns:
        Dictionary with execution results
    """
    function_name = tool_call.function.name
    
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return {
            "success": False,
            "error": "Invalid function arguments JSON"
        }
        
    try:
        if function_name == "execute_python_code":
            code = arguments["code"]
            language = arguments.get("language", "python")
            
            result = runner.execute(code, language=language)
            
            return {
                "success": True,
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "execution_time": result["execution_time"],
                "output": result["stdout"]  # For backward compatibility
            }
            
        elif function_name == "execute_bash_command":
            command = arguments["command"]
            
            result = runner.execute(command, language="bash")
            
            return {
                "success": True,
                "stdout": result["stdout"], 
                "stderr": result["stderr"],
                "execution_time": result["execution_time"],
                "output": result["stdout"]
            }
            
        else:
            return {
                "success": False,
                "error": f"Unknown function: {function_name}"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Execution failed: {str(e)}",
            "stderr": str(e)
        }

# Example usage in docstring
__doc__ = """
Example usage with OpenAI:

```python
from openai import OpenAI
from coderunner import CodeRunner
from coderunner.integrations.openai import get_tools, execute_tool

# Setup (works the same with cloud: from coderunner.cloud import InstaVM as CodeRunner)
runner = CodeRunner()
client = OpenAI(api_key="your-openai-key")

# Get tools
tools = get_tools()

# Chat with function calling
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Calculate the square root of 144"}],
    tools=tools,
    tool_choice="auto"
)

# Execute any tool calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        result = execute_tool(runner, tool_call)
        print(f"Result: {result}")
```
"""
```

**File: `/coderunner/integrations/langchain.py`**
```python
"""LangChain integration for CodeRunner"""

try:
    from langchain.tools import BaseTool
    from langchain.pydantic_v1 import BaseModel, Field
    
    class CodeExecutionInput(BaseModel):
        code: str = Field(description="Python code to execute")
        language: str = Field(default="python", description="Programming language")
    
    class CodeRunnerTool(BaseTool):
        """LangChain tool for CodeRunner"""
        name = "code_runner"
        description = "Execute Python or bash code securely"
        args_schema = CodeExecutionInput
        
        def __init__(self, runner):
            super().__init__()
            self.runner = runner
            
        def _run(self, code: str, language: str = "python") -> str:
            try:
                result = self.runner.execute(code, language=language)
                output = result["stdout"]
                if result["stderr"]:
                    output += f"\nErrors: {result['stderr']}"
                return output
            except Exception as e:
                return f"Execution error: {str(e)}"
                
        async def _arun(self, code: str, language: str = "python") -> str:
            return self._run(code, language)
    
    def get_langchain_tool(runner):
        """Get LangChain tool for CodeRunner"""
        return CodeRunnerTool(runner)
        
except ImportError:
    def get_langchain_tool(runner):
        raise ImportError("LangChain not installed. Install with: pip install langchain")
```

#### Day 3-4: Package Distribution

**File: `setup.py`**
```python
from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Local and cloud code execution for AI applications"

setup(
    name="coderunner",
    version="1.0.0",
    description="Zero-config local code execution with seamless cloud migration",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="CodeRunner Team",
    author_email="support@coderunner.dev",
    url="https://github.com/your-org/coderunner",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "requests>=2.25.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
    ],
    extras_require={
        "cloud": ["instavm>=1.0.0"],
        "integrations": [
            "openai>=1.0.0", 
            "langchain>=0.1.0",
            "llamaindex>=0.1.0"
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.18.0",
            "black>=21.0.0",
            "isort>=5.0.0",
            "mypy>=0.900"
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="code execution, ai, jupyter, docker, local development, cloud computing",
    entry_points={
        "console_scripts": [
            "coderunner=coderunner.cli:main",
        ],
    },
)
```

**File: `README.md`**
```markdown
# CodeRunner

Zero-configuration local code execution with seamless cloud migration.

## Quick Start

```python
# Install
pip install coderunner

# Use immediately - no setup required!
from coderunner import CodeRunner

runner = CodeRunner()  # Auto-starts local container
result = runner.execute("print('Hello World!')")
print(result['stdout'])  # "Hello World!"
```

## Cloud Migration (1-2 lines)

When ready for cloud scale:

```python
# Just change the import and add API key
from coderunner.cloud import InstaVM as CodeRunner

runner = CodeRunner(api_key="your-key")  # Same interface!
result = runner.execute("print('Hello Cloud!')")
```

## Features

- 🚀 **Zero Config**: No accounts, API keys, or setup required
- 🔒 **Secure**: Docker-based isolation
- 🌐 **Cloud Ready**: Seamless migration to cloud execution
- 🤖 **AI Friendly**: Built for AI agents and LLM applications
- 🔧 **Multi-Language**: Python, bash, shell support
- 📊 **Rich Output**: Captures stdout, stderr, execution time

## AI Integration

Works with OpenAI, LangChain, LlamaIndex and more:

```python
from coderunner.integrations.openai import get_tools, execute_tool

tools = get_tools()
# Use with OpenAI function calling...
```
```

#### Day 5: Testing & Documentation

**File: `tests/test_coderunner.py`**
```python
import pytest
from coderunner import CodeRunner

class TestCodeRunner:
    def test_basic_execution(self):
        runner = CodeRunner()
        result = runner.execute("print('hello')")
        assert result['stdout'].strip() == 'hello'
        
    def test_session_management(self):
        runner = CodeRunner()
        session_id = runner.start_session()
        assert session_id is not None
        assert runner.is_session_active()
        
    def test_multi_language(self):
        runner = CodeRunner()
        result = runner.execute("echo 'hello'", language="bash")
        assert 'hello' in result['stdout']
        
    def test_context_manager(self):
        with CodeRunner() as runner:
            result = runner.execute("x = 5")
            result = runner.execute("print(x)")
            assert '5' in result['stdout']
```

## Phase 2: Browser Integration Implementation (Weeks 4-6)

### Week 4: Browser Session Management

#### Enhanced Browser Infrastructure

**File: `/coderunner/browser/session_manager.py`**
```python
from dataclasses import dataclass
from typing import Dict, Optional, List
from playwright.async_api import Browser, BrowserContext, Page, async_playwright
import uuid
import asyncio

@dataclass  
class BrowserSession:
    id: str
    browser: Browser
    context: BrowserContext
    page: Page
    width: int = 1920
    height: int = 1080
    user_agent: Optional[str] = None

class BrowserSessionManager:
    """Browser session manager matching InstaVM interface exactly"""
    
    def __init__(self):
        self.sessions: Dict[str, BrowserSession] = {}
        self.playwright = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize Playwright"""
        if not self._initialized:
            self.playwright = await async_playwright().start()
            self._initialized = True
        
    async def create_session(self, width: int = 1920, height: int = 1080, 
                           user_agent: str = None) -> str:
        """Create browser session - InstaVM compatible interface"""
        await self.initialize()
        
        session_id = str(uuid.uuid4())
        
        # Launch browser with InstaVM-compatible settings
        browser = await self.playwright.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )
        
        context = await browser.new_context(
            viewport={'width': width, 'height': height},
            user_agent=user_agent,
            ignore_https_errors=True
        )
        
        page = await context.new_page()
        
        session = BrowserSession(
            id=session_id,
            browser=browser,
            context=context, 
            page=page,
            width=width,
            height=height,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        return session_id
        
    async def get_session(self, session_id: str) -> Optional[BrowserSession]:
        """Get browser session by ID"""
        return self.sessions.get(session_id)
        
    async def close_session(self, session_id: str) -> bool:
        """Close browser session"""
        session = self.sessions.get(session_id)
        if session:
            await session.browser.close()
            del self.sessions[session_id]
            return True
        return False
        
    async def list_sessions(self) -> List[Dict]:
        """List active browser sessions - InstaVM compatible"""
        return [
            {
                "session_id": session.id,
                "viewport": {"width": session.width, "height": session.height},
                "user_agent": session.user_agent
            }
            for session in self.sessions.values()
        ]
        
    # Browser interaction methods matching InstaVM API exactly
    
    async def navigate(self, session_id: str, url: str, wait_timeout: int = 30000) -> Dict:
        """Navigate to URL - InstaVM compatible"""
        session = await self.get_session(session_id)
        if not session:
            raise BrowserSessionError("Session not found")
            
        try:
            await session.page.goto(url, timeout=wait_timeout)
            return {
                "success": True,
                "url": session.page.url,
                "title": await session.page.title(),
                "status": "completed"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": "failed"
            }
            
    async def click(self, session_id: str, selector: str, force: bool = False, 
                   timeout: int = 30000) -> Dict:
        """Click element - InstaVM compatible"""
        session = await self.get_session(session_id)
        if not session:
            raise BrowserSessionError("Session not found")
            
        try:
            await session.page.click(selector, force=force, timeout=timeout)
            return {
                "success": True,
                "selector": selector,
                "status": "completed"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "selector": selector,
                "status": "failed"
            }
            
    async def type_text(self, session_id: str, selector: str, text: str, 
                       delay: int = 100, timeout: int = 30000) -> Dict:
        """Type text into element - InstaVM compatible"""
        session = await self.get_session(session_id)
        if not session:
            raise BrowserSessionError("Session not found")
            
        try:
            await session.page.type(selector, text, delay=delay, timeout=timeout)
            return {
                "success": True,
                "selector": selector,
                "text": text,
                "status": "completed"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "selector": selector,
                "status": "failed"
            }
            
    async def fill(self, session_id: str, selector: str, value: str, 
                  timeout: int = 30000) -> Dict:
        """Fill form field - InstaVM compatible"""
        session = await self.get_session(session_id)
        if not session:
            raise BrowserSessionError("Session not found")
            
        try:
            await session.page.fill(selector, value, timeout=timeout)
            return {
                "success": True,
                "selector": selector,
                "value": value,
                "status": "completed"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "selector": selector,
                "status": "failed"
            }
            
    async def scroll(self, session_id: str, selector: str = None, 
                    x: int = None, y: int = None) -> Dict:
        """Scroll page or element - InstaVM compatible"""
        session = await self.get_session(session_id)
        if not session:
            raise BrowserSessionError("Session not found")
            
        try:
            if selector:
                # Scroll element into view
                await session.page.locator(selector).scroll_into_view_if_needed()
            elif x is not None or y is not None:
                # Scroll to coordinates
                await session.page.evaluate(f"window.scrollTo({x or 0}, {y or 0})")
            else:
                # Scroll down by default
                await session.page.evaluate("window.scrollBy(0, 500)")
                
            return {
                "success": True,
                "status": "completed"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": "failed"
            }
            
    async def wait_for(self, session_id: str, condition: str, selector: str = None, 
                      timeout: int = 30000) -> Dict:
        """Wait for condition - InstaVM compatible"""
        session = await self.get_session(session_id)
        if not session:
            raise BrowserSessionError("Session not found")
            
        try:
            if condition == "visible" and selector:
                await session.page.wait_for_selector(selector, state="visible", timeout=timeout)
            elif condition == "hidden" and selector:
                await session.page.wait_for_selector(selector, state="hidden", timeout=timeout)
            elif condition == "load":
                await session.page.wait_for_load_state("load", timeout=timeout)
            elif condition == "networkidle":
                await session.page.wait_for_load_state("networkidle", timeout=timeout)
            else:
                raise ValueError(f"Unknown condition: {condition}")
                
            return {
                "success": True,
                "condition": condition,
                "selector": selector,
                "status": "completed"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "condition": condition,
                "status": "failed"
            }
            
    async def screenshot(self, session_id: str, full_page: bool = True, 
                        clip: Dict = None, format: str = "png", 
                        quality: int = None) -> str:
        """Take screenshot - InstaVM compatible"""
        session = await self.get_session(session_id)
        if not session:
            raise BrowserSessionError("Session not found")
            
        screenshot_options = {
            "full_page": full_page,
            "type": format
        }
        
        if clip:
            screenshot_options["clip"] = clip
        if quality and format == "jpeg":
            screenshot_options["quality"] = quality
            
        screenshot_bytes = await session.page.screenshot(**screenshot_options)
        
        # Return base64 encoded like InstaVM
        import base64
        return base64.b64encode(screenshot_bytes).decode()
        
    async def extract_elements(self, session_id: str, selector: str = None, 
                             attributes: List[str] = None) -> List[Dict]:
        """Extract DOM elements - InstaVM compatible"""
        session = await self.get_session(session_id)
        if not session:
            raise BrowserSessionError("Session not found")
            
        if not selector:
            selector = "body *"
        if not attributes:
            attributes = ["text", "href", "src", "class", "id"]
            
        # Extract elements using JavaScript
        js_code = f"""
        Array.from(document.querySelectorAll('{selector}')).slice(0, 100).map(el => {{
            const result = {{}};
            {chr(10).join([f"result['{attr}'] = el.{attr} || el.getAttribute('{attr}') || '';" for attr in attributes])}
            result.tagName = el.tagName.toLowerCase();
            result.text = el.textContent ? el.textContent.trim() : '';
            return result;
        }})
        """
        
        elements = await session.page.evaluate(js_code)
        return elements

# Global browser session manager
browser_session_manager = BrowserSessionManager()

class BrowserSessionError(Exception):
    pass
```

### Week 5: REST API Browser Endpoints

**File: `/coderunner/api/browser_routes.py`**
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from ..browser.session_manager import browser_session_manager, BrowserSessionError

router = APIRouter(prefix="/v1/browser")

# Request/Response models matching InstaVM exactly

class BrowserCreateRequest(BaseModel):
    viewport_width: int = 1920
    viewport_height: int = 1080
    user_agent: Optional[str] = None

class BrowserCreateResponse(BaseModel):
    session_id: str

class NavigateRequest(BaseModel):
    url: str
    session_id: str
    wait_timeout: int = 30000

class ClickRequest(BaseModel):
    selector: str
    session_id: str
    force: bool = False
    timeout: int = 30000

class TypeRequest(BaseModel):
    selector: str
    text: str
    session_id: str
    delay: int = 100
    timeout: int = 30000

class FillRequest(BaseModel):
    selector: str
    value: str
    session_id: str
    timeout: int = 30000

class ScrollRequest(BaseModel):
    session_id: str
    selector: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None

class WaitRequest(BaseModel):
    condition: str
    session_id: str
    selector: Optional[str] = None
    timeout: int = 30000

class ScreenshotRequest(BaseModel):
    session_id: str
    full_page: bool = True
    clip: Optional[Dict] = None
    format: str = "png"
    quality: Optional[int] = None

class ExtractRequest(BaseModel):
    session_id: str
    selector: Optional[str] = None
    attributes: Optional[List[str]] = None

# Browser session endpoints

@router.post("/sessions/", response_model=BrowserCreateResponse)
async def create_browser_session(request: BrowserCreateRequest):
    """Create new browser session - InstaVM compatible"""
    try:
        session_id = await browser_session_manager.create_session(
            width=request.viewport_width,
            height=request.viewport_height,
            user_agent=request.user_agent
        )
        return BrowserCreateResponse(session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create browser session: {str(e)}")

@router.get("/sessions/{session_id}")
async def get_browser_session(session_id: str):
    """Get browser session info - InstaVM compatible"""
    session = await browser_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Browser session not found")
        
    return {
        "session_id": session.id,
        "viewport": {"width": session.width, "height": session.height},
        "user_agent": session.user_agent,
        "status": "active"
    }

@router.delete("/sessions/{session_id}")
async def close_browser_session(session_id: str):
    """Close browser session - InstaVM compatible"""
    success = await browser_session_manager.close_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Browser session not found")
    return {"success": True}

@router.get("/sessions/")
async def list_browser_sessions():
    """List active browser sessions - InstaVM compatible"""
    sessions = await browser_session_manager.list_sessions()
    return {"sessions": sessions}

# Browser interaction endpoints

@router.post("/interactions/navigate")
async def navigate(request: NavigateRequest):
    """Navigate to URL - InstaVM compatible"""
    try:
        result = await browser_session_manager.navigate(
            request.session_id, 
            request.url, 
            request.wait_timeout
        )
        return result
    except BrowserSessionError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/interactions/click")  
async def click_element(request: ClickRequest):
    """Click element - InstaVM compatible"""
    try:
        result = await browser_session_manager.click(
            request.session_id,
            request.selector,
            request.force,
            request.timeout
        )
        return result
    except BrowserSessionError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/interactions/type")
async def type_text(request: TypeRequest):
    """Type text into element - InstaVM compatible"""
    try:
        result = await browser_session_manager.type_text(
            request.session_id,
            request.selector,
            request.text,
            request.delay,
            request.timeout
        )
        return result
    except BrowserSessionError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/interactions/fill")
async def fill_form(request: FillRequest):
    """Fill form field - InstaVM compatible"""
    try:
        result = await browser_session_manager.fill(
            request.session_id,
            request.selector,
            request.value,
            request.timeout
        )
        return result
    except BrowserSessionError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/interactions/scroll")
async def scroll_page(request: ScrollRequest):
    """Scroll page or element - InstaVM compatible"""
    try:
        result = await browser_session_manager.scroll(
            request.session_id,
            request.selector,
            request.x,
            request.y
        )
        return result
    except BrowserSessionError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/interactions/wait")
async def wait_for_condition(request: WaitRequest):
    """Wait for condition - InstaVM compatible"""
    try:
        result = await browser_session_manager.wait_for(
            request.session_id,
            request.condition,
            request.selector,
            request.timeout
        )
        return result
    except BrowserSessionError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/interactions/screenshot")
async def take_screenshot(request: ScreenshotRequest):
    """Take screenshot - InstaVM compatible"""
    try:
        screenshot = await browser_session_manager.screenshot(
            request.session_id,
            request.full_page,
            request.clip,
            request.format,
            request.quality
        )
        return {"screenshot": screenshot}
    except BrowserSessionError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/interactions/extract")
async def extract_elements(request: ExtractRequest):
    """Extract DOM elements - InstaVM compatible"""
    try:
        elements = await browser_session_manager.extract_elements(
            request.session_id,
            request.selector,
            request.attributes
        )
        return {"elements": elements}
    except BrowserSessionError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Week 6: Client Browser Integration

**File: `/coderunner/browser/client.py`**
```python
"""Browser client classes matching InstaVM interface exactly"""

from typing import Dict, Any, List, Optional
import requests

class BrowserSession:
    """Browser session matching InstaVM interface exactly"""
    
    def __init__(self, session_id: str, runner: 'CodeRunner'):
        self.session_id = session_id
        self.runner = runner
        self._active = True
        
    def navigate(self, url: str, wait_timeout: int = 30000) -> Dict[str, Any]:
        """Navigate to URL - InstaVM compatible"""
        if not self._active:
            raise BrowserSessionError("Browser session is closed")
            
        return self.runner._browser_request("navigate", {
            "session_id": self.session_id,
            "url": url, 
            "wait_timeout": wait_timeout
        })
        
    def click(self, selector: str, force: bool = False, timeout: int = 30000) -> Dict[str, Any]:
        """Click element by CSS selector - InstaVM compatible"""
        if not self._active:
            raise BrowserSessionError("Browser session is closed")
            
        return self.runner._browser_request("click", {
            "session_id": self.session_id,
            "selector": selector,
            "force": force,
            "timeout": timeout
        })
        
    def type(self, selector: str, text: str, delay: int = 100, timeout: int = 30000) -> Dict[str, Any]:
        """Type text into element - InstaVM compatible"""
        if not self._active:
            raise BrowserSessionError("Browser session is closed")
            
        return self.runner._browser_request("type", {
            "session_id": self.session_id,
            "selector": selector,
            "text": text,
            "delay": delay,
            "timeout": timeout
        })
        
    def fill(self, selector: str, value: str, timeout: int = 30000) -> Dict[str, Any]:
        """Fill form field - InstaVM compatible"""
        if not self._active:
            raise BrowserSessionError("Browser session is closed")
            
        return self.runner._browser_request("fill", {
            "session_id": self.session_id,
            "selector": selector,
            "value": value,
            "timeout": timeout
        })
        
    def scroll(self, selector: str = None, x: int = None, y: int = None) -> Dict[str, Any]:
        """Scroll page or element - InstaVM compatible"""
        if not self._active:
            raise BrowserSessionError("Browser session is closed")
            
        return self.runner._browser_request("scroll", {
            "session_id": self.session_id,
            "selector": selector,
            "x": x,
            "y": y
        })
        
    def wait_for(self, condition: str, selector: str = None, timeout: int = 30000) -> Dict[str, Any]:
        """Wait for condition - InstaVM compatible"""
        if not self._active:
            raise BrowserSessionError("Browser session is closed")
            
        return self.runner._browser_request("wait", {
            "session_id": self.session_id,
            "condition": condition,
            "selector": selector,
            "timeout": timeout
        })
        
    def screenshot(self, full_page: bool = True, clip: Dict = None, 
                  format: str = "png", quality: int = None) -> str:
        """Take screenshot (returns base64 string) - InstaVM compatible"""
        if not self._active:
            raise BrowserSessionError("Browser session is closed")
            
        result = self.runner._browser_request("screenshot", {
            "session_id": self.session_id,
            "full_page": full_page,
            "clip": clip,
            "format": format,
            "quality": quality
        })
        return result["screenshot"]
        
    def extract_elements(self, selector: str = None, attributes: List[str] = None) -> List[Dict[str, Any]]:
        """Extract DOM elements - InstaVM compatible"""
        if not self._active:
            raise BrowserSessionError("Browser session is closed")
            
        result = self.runner._browser_request("extract", {
            "session_id": self.session_id,
            "selector": selector,
            "attributes": attributes
        })
        return result["elements"]
        
    def close(self) -> bool:
        """Close browser session - InstaVM compatible"""
        if self._active:
            try:
                response = self.runner._session.delete(
                    f"{self.runner.base_url}/v1/browser/sessions/{self.session_id}"
                )
                self._active = False
                return response.status_code == 200
            except:
                self._active = False
                return False
        return True
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class BrowserManager:
    """Browser manager matching InstaVM interface exactly"""
    
    def __init__(self, runner: 'CodeRunner'):
        self.runner = runner
        self._default_session_id = None
        
    def create_session(self, viewport_width: int = 1920, viewport_height: int = 1080, 
                      user_agent: str = None) -> BrowserSession:
        """Create browser session - InstaVM compatible"""
        response = self.runner._session.post(
            f"{self.runner.base_url}/v1/browser/sessions/", 
            json={
                "viewport_width": viewport_width,
                "viewport_height": viewport_height,
                "user_agent": user_agent
            }
        )
        
        if response.status_code != 200:
            raise BrowserSessionError(f"Failed to create browser session: {response.text}")
            
        result = response.json()
        return BrowserSession(result["session_id"], self.runner)
        
    def _ensure_default_session(self) -> str:
        """Ensure a default browser session exists"""
        if not self._default_session_id:
            session = self.create_session()
            self._default_session_id = session.session_id
        return self._default_session_id
        
    # Convenience methods that auto-create session if needed
    
    def navigate(self, url: str, session_id: str = None, wait_timeout: int = 30000) -> Dict[str, Any]:
        """Navigate to URL (auto-creates session if none provided)"""
        if not session_id:
            session_id = self._ensure_default_session()
        return self.runner._browser_request("navigate", {
            "url": url,
            "session_id": session_id,
            "wait_timeout": wait_timeout
        })
        
    def click(self, selector: str, session_id: str = None, force: bool = False, timeout: int = 30000) -> Dict[str, Any]:
        """Click element by CSS selector"""
        if not session_id:
            session_id = self._ensure_default_session()
        return self.runner._browser_request("click", {
            "selector": selector,
            "session_id": session_id,
            "force": force,
            "timeout": timeout
        })
        
    def screenshot(self, session_id: str = None, full_page: bool = True, clip: Dict = None, 
                  format: str = "png", quality: int = None) -> str:
        """Take screenshot (returns base64)"""
        if not session_id:
            session_id = self._ensure_default_session()
        result = self.runner._browser_request("screenshot", {
            "session_id": session_id,
            "full_page": full_page,
            "clip": clip,
            "format": format,
            "quality": quality
        })
        return result["screenshot"]
        
    def extract_elements(self, selector: str = None, session_id: str = None, 
                        attributes: List[str] = None) -> List[Dict]:
        """Extract DOM elements"""
        if not session_id:
            session_id = self._ensure_default_session()
        result = self.runner._browser_request("extract", {
            "session_id": session_id,
            "selector": selector,
            "attributes": attributes
        })
        return result["elements"]
        
    def close(self):
        """Close default browser session if one was created"""
        if self._default_session_id:
            try:
                self.runner._session.delete(
                    f"{self.runner.base_url}/v1/browser/sessions/{self._default_session_id}"
                )
            except:
                pass
            self._default_session_id = None

class BrowserSessionError(Exception):
    """Browser session error"""
    pass
```

**Update `/coderunner/__init__.py`** to include browser functionality:

```python
# Add to CodeRunner class:

class CodeRunner:
    def __init__(self, auto_start: bool = True):
        # ... existing code ...
        
        # Add browser manager
        self.browser = BrowserManager(self)
        
    def create_browser_session(self, viewport_width: int = 1920, viewport_height: int = 1080, 
                              user_agent: str = None) -> 'BrowserSession':
        """Create browser session - InstaVM compatible interface"""
        from .browser.client import BrowserSession
        
        response = self._session.post(f"{self.base_url}/v1/browser/sessions/", json={
            "viewport_width": viewport_width,
            "viewport_height": viewport_height,
            "user_agent": user_agent
        })
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to create browser session: {response.text}")
            
        result = response.json()
        return BrowserSession(result["session_id"], self)
        
    def _browser_request(self, action: str, data: dict) -> dict:
        """Internal browser API request helper"""
        try:
            response = self._session.post(
                f"{self.base_url}/v1/browser/interactions/{action}", 
                json=data
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Browser {action} failed: {response.text}")
                
            return response.json()
            
        except requests.exceptions.ConnectionError:
            raise CodeRunnerError("Cannot connect to CodeRunner browser service")
```

## Enhanced OpenAI Integration with Browser

**Update `/coderunner/integrations/openai.py`:**

```python
def get_tools() -> List[Dict[str, Any]]:
    """Get OpenAI function calling tools including browser automation"""
    return [
        # ... existing execution tools ...
        
        {
            "type": "function",
            "function": {
                "name": "create_browser_session",
                "description": "Create a new browser session for web automation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "width": {"type": "integer", "description": "Browser width", "default": 1920},
                        "height": {"type": "integer", "description": "Browser height", "default": 1080}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "navigate_to_url", 
                "description": "Navigate browser to URL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to navigate to"}
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "extract_page_content",
                "description": "Extract text content from page",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "description": "CSS selector", "default": "body"}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "click_element",
                "description": "Click a page element",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "description": "CSS selector"}
                    },
                    "required": ["selector"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "take_screenshot",
                "description": "Take page screenshot", 
                "parameters": {
                    "type": "object",
                    "properties": {
                        "full_page": {"type": "boolean", "description": "Full page capture", "default": True}
                    }
                }
            }
        }
    ]

def execute_tool(runner, tool_call, browser_session=None):
    """Execute OpenAI tool call - works with both local and cloud"""
    function_name = tool_call.function.name
    
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        arguments = {}
        
    try:
        # Code execution tools
        if function_name in ["execute_python_code", "execute_bash_command"]:
            # ... existing code execution logic ...
            pass
            
        # Browser automation tools
        elif function_name == "create_browser_session":
            width = arguments.get("width", 1920)
            height = arguments.get("height", 1080)
            session = runner.browser.create_session(width, height)
            return {
                "success": True,
                "session_id": session.session_id,
                "session": session,
                "message": f"Created browser session {session.session_id}"
            }
            
        elif function_name == "navigate_to_url":
            if not browser_session:
                return {"success": False, "error": "No browser session. Create one first."}
            url = arguments["url"]
            result = browser_session.navigate(url)
            return {"success": True, "message": f"Navigated to {url}", "result": result}
            
        elif function_name == "extract_page_content":
            if not browser_session:
                return {"success": False, "error": "No browser session active"}
            selector = arguments.get("selector", "body")
            elements = browser_session.extract_elements(selector, ["text"])
            if elements:
                content = elements[0].get("text", "")
                return {"success": True, "content": content, "length": len(content)}
            return {"success": False, "error": "No content found"}
            
        elif function_name == "click_element":
            if not browser_session:
                return {"success": False, "error": "No browser session active"}
            selector = arguments["selector"]
            result = browser_session.click(selector)
            return {"success": True, "message": f"Clicked {selector}", "result": result}
            
        elif function_name == "take_screenshot":
            if not browser_session:
                return {"success": False, "error": "No browser session active"}
            full_page = arguments.get("full_page", True)
            screenshot = browser_session.screenshot(full_page=full_page)
            return {"success": True, "screenshot_length": len(screenshot), "screenshot": screenshot}
            
        else:
            return {"success": False, "error": f"Unknown function: {function_name}"}
            
    except Exception as e:
        return {"success": False, "error": f"Function {function_name} failed: {str(e)}"}
```

## Usage Examples

### Basic Code Execution
```python
from coderunner import CodeRunner

# Local execution (auto-starts container)
runner = CodeRunner()

# Execute Python
result = runner.execute("print('Hello Local!')")
print(result['stdout'])  # "Hello Local!"

# Execute bash
result = runner.execute("echo 'Hello Bash!'", language="bash")
print(result['stdout'])  # "Hello Bash!"
```

### Browser Automation
```python
from coderunner import CodeRunner

runner = CodeRunner()

# Create browser session
browser = runner.browser.create_session()

# Navigate and interact
browser.navigate("https://httpbin.org")
browser.click("a[href='/get']")
screenshot = browser.screenshot()

# Extract data
elements = browser.extract_elements("h1", ["text"])
print(elements)  # [{"text": "httpbin.org", "tagName": "h1", ...}]

browser.close()
```

### Cloud Migration
```python
# Option A: Import alias (recommended)
from coderunner.cloud import InstaVM as CodeRunner

runner = CodeRunner(api_key="your-key")  # Only change needed!

# Everything else works exactly the same
result = runner.execute("print('Hello Cloud!')")

browser = runner.browser.create_session()
browser.navigate("https://example.com")
```

### OpenAI Integration (Works with Both)
```python
from openai import OpenAI
from coderunner import CodeRunner  # or from coderunner.cloud import InstaVM as CodeRunner
from coderunner.integrations.openai import get_tools, execute_tool

# Setup
runner = CodeRunner()  # or CodeRunner(api_key="key") for cloud
client = OpenAI(api_key="openai-key")

# Get tools
tools = get_tools()

# Use with AI
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Go to httpbin.org and show me my IP"}],
    tools=tools,
    tool_choice="auto"
)

# Execute any tool calls
browser_session = None
for tool_call in response.choices[0].message.tool_calls:
    result = execute_tool(runner, tool_call, browser_session)
    if result.get("session"):
        browser_session = result["session"]
    print(result)
```

## Migration Strategy Summary

### For New Users (Local-First)
1. `pip install coderunner`
2. `from coderunner import CodeRunner`
3. `runner = CodeRunner()` - Auto-starts, no config needed
4. Use immediately with full Python, bash, and browser automation

### For InstaVM Users (Cloud-First) 
1. `pip install coderunner` - Test locally
2. Change import: `from coderunner.cloud import InstaVM as CodeRunner`
3. Add API key: `runner = CodeRunner(api_key="key")`
4. Everything else stays exactly the same

### Key Benefits
- **Zero friction** for new users (no signups, API keys)
- **1-2 line migration** between local and cloud  
- **Identical interfaces** - same code works everywhere
- **Gradual adoption** - existing InstaVM users can test locally
- **Hybrid workflows** - local development, cloud production

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "Create detailed implementation plan for code execution part", "status": "completed"}, {"id": "2", "content": "Design browser integration phase", "status": "completed"}, {"id": "3", "content": "Document everything in parity.md for future reference", "status": "completed"}]