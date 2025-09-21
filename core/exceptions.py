"""Core exceptions for CodeRunner"""


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


class KernelError(CodeRunnerError):
    """Kernel-related errors"""
    pass


class LanguageError(CodeRunnerError):
    """Language processing errors"""
    pass