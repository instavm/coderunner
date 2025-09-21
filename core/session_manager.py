"""Session management for CodeRunner execution contexts"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime, timedelta
import uuid
import asyncio
import logging

from .exceptions import SessionError
from .language_processor import LanguageProcessor

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Execution session data"""
    id: str
    kernel_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    language: str = "python"
    variables: Dict = field(default_factory=dict)
    
    def touch(self):
        """Update last_used timestamp"""
        self.last_used = datetime.now()
        
    def is_expired(self, timeout_hours: int = 24) -> bool:
        """Check if session has expired"""
        cutoff = datetime.now() - timedelta(hours=timeout_hours)
        return self.last_used < cutoff


class SessionManager:
    """Manage execution sessions with automatic cleanup"""
    
    def __init__(self, session_timeout_hours: int = 24):
        self.sessions: Dict[str, Session] = {}
        self.session_timeout_hours = session_timeout_hours
        self._cleanup_task: Optional[asyncio.Task] = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize session manager and start background tasks"""
        if self._initialized:
            return
            
        logger.info("Initializing session manager")
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._initialized = True
        
    async def shutdown(self):
        """Shutdown session manager and cleanup resources"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
        # Close all active sessions
        for session_id in list(self.sessions.keys()):
            await self.close_session(session_id)
            
        logger.info("Session manager shutdown complete")
        
    async def create_session(self, language: str = "python") -> str:
        """
        Create new execution session with dedicated kernel
        
        Args:
            language: Programming language for the session
            
        Returns:
            Session ID string
        """
        # Normalize language
        language = LanguageProcessor.normalize_language(language)
        
        session_id = str(uuid.uuid4())
        
        # Import here to avoid circular imports
        from core.kernel_pool import kernel_pool
        
        # Get dedicated kernel from existing pool
        try:
            kernel_id = await kernel_pool.get_available_kernel()
            if not kernel_id:
                raise SessionError("No available kernels for new session")
        except Exception as e:
            logger.error(f"Failed to get kernel for session: {e}")
            raise SessionError(f"Failed to allocate kernel: {str(e)}")
        
        session = Session(
            id=session_id,
            kernel_id=kernel_id,
            language=language
        )
        
        self.sessions[session_id] = session
        logger.info(f"Created session {session_id} with kernel {kernel_id}")
        
        return session_id
        
    async def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get session by ID
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session object or None if not found
        """
        session = self.sessions.get(session_id)
        if session:
            session.touch()
        return session
        
    async def execute_in_session(self, session_id: str, code: str) -> dict:
        """
        Execute code in specific session
        
        Args:
            session_id: Session identifier
            code: Code to execute
            
        Returns:
            Execution result dictionary
        """
        session = self.sessions.get(session_id)
        if not session:
            raise SessionError(f"Session {session_id} not found")
        
        session.touch()
        
        # Preprocess command based on session language
        processed_code = LanguageProcessor.preprocess_command(code, session.language)
        
        # Import here to avoid circular imports
        from core.kernel_pool import kernel_pool
        
        try:
            # Execute using existing kernel pool with specific kernel
            result = await kernel_pool.execute_on_kernel(session.kernel_id, processed_code)
            
            # Store any variables for session context (if needed)
            if isinstance(result, dict) and "variables" in result:
                session.variables.update(result["variables"])
                
            return result
            
        except Exception as e:
            logger.error(f"Execution failed in session {session_id}: {e}")
            raise SessionError(f"Execution failed: {str(e)}")
        
    async def close_session(self, session_id: str) -> bool:
        """
        Close session and release resources
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was closed, False if not found
        """
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        try:
            # Import here to avoid circular imports
            from server import kernel_pool
            
            # Release kernel back to pool
            await kernel_pool.release_kernel(session.kernel_id)
            
        except Exception as e:
            logger.warning(f"Error releasing kernel for session {session_id}: {e}")
        
        # Remove session
        del self.sessions[session_id]
        logger.info(f"Closed session {session_id}")
        
        return True
        
    async def list_sessions(self) -> Dict[str, dict]:
        """
        List all active sessions
        
        Returns:
            Dictionary of session data
        """
        result = {}
        for session_id, session in self.sessions.items():
            result[session_id] = {
                "id": session.id,
                "language": session.language,
                "created_at": session.created_at.isoformat(),
                "last_used": session.last_used.isoformat(),
                "kernel_id": session.kernel_id
            }
        return result
        
    async def _cleanup_loop(self):
        """Background task to cleanup expired sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                expired_sessions = [
                    session_id for session_id, session in self.sessions.items()
                    if session.is_expired(self.session_timeout_hours)
                ]
                
                for session_id in expired_sessions:
                    logger.info(f"Cleaning up expired session: {session_id}")
                    await self.close_session(session_id)
                    
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup loop: {e}")


# Global session manager instance
session_manager = SessionManager()