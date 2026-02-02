"""
CodeRunner - Sandboxed Python execution for Mac.

Usage:
    from coderunner import CodeRunner, execute

    # Using the client
    cr = CodeRunner()
    result = cr.execute("print('hello')")
    print(result.stdout)

    # One-liner
    print(execute("2 + 2"))
"""

from .client import CodeRunner, ExecutionResult, BrowserResult, execute

__all__ = ["CodeRunner", "ExecutionResult", "BrowserResult", "execute"]
__version__ = "0.1.0"
