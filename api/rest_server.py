"""REST API server for CodeRunner - InstaVM compatible interface"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
import uuid
from typing import Dict, Any

from .schemas.execution import (
    CommandRequest, ExecutionResponse, AsyncExecutionResponse,
    SessionResponse, HealthResponse
)
from ..core.session_manager import session_manager, SessionError
from ..core.language_processor import LanguageProcessor
from ..core.exceptions import ExecutionError, KernelError

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CodeRunner REST API",
    description="Local code execution with InstaVM-compatible interface",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Async task storage (simple in-memory for now)
# TODO: Replace with persistent storage (Redis/database) for production to survive server restarts
async_tasks: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        await session_manager.initialize()
        logger.info("REST API server started successfully")
    except Exception as e:
        logger.error(f"Failed to start REST API server: {e}")
        raise


@app.on_event("shutdown")  
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        await session_manager.shutdown()
        logger.info("REST API server shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check session manager
        session_count = len(session_manager.sessions)
        
        # Check kernel pool
        from ..core.kernel_pool import kernel_pool
        kernel_status = {
            "total_kernels": len(kernel_pool.kernels),
            "available_kernels": len([k for k in kernel_pool.kernels.values() if k.is_available()]),
            "busy_kernels": len(kernel_pool.busy_kernels)
        }
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            services={
                "session_manager": {"active_sessions": session_count},
                "kernel_pool": kernel_status
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/execute", response_model=ExecutionResponse)
async def execute_command(request: CommandRequest):
    """
    Execute command synchronously - InstaVM compatible interface
    
    Args:
        request: Command execution request
        
    Returns:
        Execution results with stdout, stderr, timing
    """
    start_time = time.time()
    cpu_start_time = time.process_time()
    
    try:
        # Validate language
        if not LanguageProcessor.validate_language(request.language):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported language: {request.language}"
            )
        
        # Get or create session
        session_id = request.session_id
        if not session_id:
            session_id = await session_manager.create_session(request.language)
        elif not await session_manager.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Execute command
        try:
            result = await session_manager.execute_in_session(session_id, request.command)
        except SessionError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")
        
        # Calculate timing
        execution_time = time.time() - start_time
        cpu_time = time.process_time() - cpu_start_time
        
        # Extract output from result
        stdout = ""
        stderr = ""
        
        if isinstance(result, dict):
            # Handle different result formats from kernel pool
            if "stdout" in result:
                stdout = result["stdout"]
            elif "output" in result:
                stdout = result["output"]
            else:
                stdout = str(result)
                
            if "stderr" in result:
                stderr = result["stderr"]
            elif "error" in result and result.get("error"):
                stderr = str(result["error"])
        else:
            stdout = str(result)
        
        return ExecutionResponse(
            stdout=stdout,
            stderr=stderr,
            execution_time=execution_time,
            cpu_time=cpu_time,
            session_id=session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in execute_command: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def _execute_task_background(task_id: str, command: str, language: str, session_id: str):
    """Background task to execute code asynchronously"""
    try:
        # Update status to running
        async_tasks[task_id]["status"] = "running"
        
        # Get or create session
        if not session_id:
            session_id = await session_manager.create_session(language)
        elif not await session_manager.get_session(session_id):
            async_tasks[task_id]["status"] = "failed"
            async_tasks[task_id]["error"] = "Session not found"
            return
        
        # Execute command
        result = await session_manager.execute_in_session(session_id, command)
        
        # Extract output from result
        stdout = ""
        stderr = ""
        
        if isinstance(result, dict):
            if "stdout" in result:
                stdout = result["stdout"]
            elif "output" in result:
                stdout = result["output"]
            else:
                stdout = str(result)
                
            if "stderr" in result:
                stderr = result["stderr"]
            elif "error" in result and result.get("error"):
                stderr = str(result["error"])
        else:
            stdout = str(result)
        
        # Update task with results
        async_tasks[task_id].update({
            "status": "completed",
            "result": {
                "stdout": stdout,
                "stderr": stderr,
                "session_id": session_id
            },
            "completed_at": time.time()
        })
        
    except Exception as e:
        logger.error(f"Background task {task_id} failed: {e}")
        async_tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": time.time()
        })


@app.post("/execute_async", response_model=AsyncExecutionResponse)
async def execute_command_async(request: CommandRequest, background_tasks: BackgroundTasks):
    """
    Execute command asynchronously - InstaVM compatible interface
    
    Args:
        request: Command execution request
        background_tasks: FastAPI background tasks
        
    Returns:
        Task ID for checking execution status
    """
    try:
        # Validate language
        if not LanguageProcessor.validate_language(request.language):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {request.language}"
            )
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Store task info
        async_tasks[task_id] = {
            "id": task_id,
            "status": "queued",
            "command": request.command,
            "language": request.language,
            "session_id": request.session_id,
            "created_at": time.time(),
            "result": None,
            "error": None
        }
        
        # Add background task for execution
        background_tasks.add_task(
            _execute_task_background,
            task_id,
            request.command,
            request.language,
            request.session_id
        )
        
        return AsyncExecutionResponse(task_id=task_id, status="queued")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in execute_command_async: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get async task status - InstaVM compatible"""
    task = async_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task


@app.post("/sessions", response_model=SessionResponse)
async def create_session(language: str = "python"):
    """
    Create new execution session
    
    Args:
        language: Programming language for session
        
    Returns:
        Session information
    """
    try:
        session_id = await session_manager.create_session(language)
        
        return SessionResponse(
            session_id=session_id,
            status="active",
            created_at=time.time()
        )
        
    except SessionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session.id,
        "language": session.language,
        "created_at": session.created_at.isoformat(),
        "last_used": session.last_used.isoformat(),
        "status": "active"
    }


@app.delete("/sessions/{session_id}")
async def close_session(session_id: str):
    """Close execution session"""
    success = await session_manager.close_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"success": True, "message": f"Session {session_id} closed"}


@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    sessions = await session_manager.list_sessions()
    return {"sessions": sessions}


@app.get("/languages")
async def get_supported_languages():
    """Get list of supported programming languages"""
    return {
        "languages": LanguageProcessor.get_supported_languages(),
        "default": "python"
    }


# Error handlers
@app.exception_handler(SessionError)
async def session_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "type": "session_error"}
    )


@app.exception_handler(ExecutionError)
async def execution_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": "execution_error"}
    )


@app.exception_handler(KernelError)
async def kernel_error_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc), "type": "kernel_error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8223)