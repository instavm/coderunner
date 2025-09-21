"""Execution API schemas - InstaVM compatible"""

from pydantic import BaseModel
from typing import Optional


class CommandRequest(BaseModel):
    """Request schema for code execution - matches InstaVM interface"""
    command: str
    language: Optional[str] = "python"
    timeout: Optional[int] = 30
    session_id: Optional[str] = None


class ExecutionResponse(BaseModel):
    """Response schema for code execution - matches InstaVM interface"""
    stdout: str
    stderr: str
    execution_time: float
    cpu_time: Optional[float] = None
    session_id: str


class AsyncExecutionResponse(BaseModel):
    """Response schema for async execution"""
    task_id: str
    status: str = "queued"


class SessionResponse(BaseModel):
    """Response schema for session operations"""
    session_id: str
    status: str = "active"
    created_at: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    services: dict = {}