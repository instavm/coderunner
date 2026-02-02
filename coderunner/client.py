"""
CodeRunner Python Client

Simple wrapper for the CodeRunner REST API.

Usage:
    from coderunner import CodeRunner

    cr = CodeRunner()  # defaults to http://coderunner.local:8222
    result = cr.execute("print('hello')")
    print(result.stdout)
"""

import httpx
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionResult:
    """Result of code execution."""
    stdout: str
    stderr: str
    execution_time: float
    success: bool


@dataclass
class BrowserResult:
    """Result of browser content extraction."""
    content: str
    url: str
    success: bool
    error: Optional[str] = None


class CodeRunner:
    """
    Python client for CodeRunner API.

    Args:
        base_url: Server URL. Defaults to http://coderunner.local:8222
        timeout: Request timeout in seconds. Defaults to 300 (5 minutes).

    Example:
        >>> cr = CodeRunner()
        >>> result = cr.execute("print('hello')")
        >>> print(result.stdout)
        hello
    """

    def __init__(
        self,
        base_url: str = "http://coderunner.local:8222",
        timeout: float = 300.0
    ):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def execute(self, code: str) -> ExecutionResult:
        """
        Execute Python code and return result.

        Args:
            code: Python code to execute.

        Returns:
            ExecutionResult with stdout, stderr, execution_time, and success.

        Example:
            >>> result = cr.execute("print(2 + 2)")
            >>> result.stdout
            '4\\n'
        """
        response = self._client.post(
            f"{self.base_url}/v1/execute",
            json={"code": code}
        )
        response.raise_for_status()
        data = response.json()
        return ExecutionResult(
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            execution_time=data.get("execution_time", 0.0),
            success=not data.get("stderr")
        )

    def browse(self, url: str) -> BrowserResult:
        """
        Navigate to URL and extract text content.

        Args:
            url: URL to navigate to.

        Returns:
            BrowserResult with extracted content.

        Example:
            >>> result = cr.browse("https://example.com")
            >>> print(result.content[:50])
        """
        response = self._client.post(
            f"{self.base_url}/v1/browser/content",
            json={"url": url}
        )
        response.raise_for_status()
        data = response.json()
        return BrowserResult(
            content=data.get("readable_content", {}).get("content", ""),
            url=url,
            success=data.get("status") == "success",
            error=data.get("error")
        )

    def health(self) -> bool:
        """
        Check if server is healthy.

        Returns:
            True if server is healthy, False otherwise.
        """
        try:
            response = self._client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except httpx.RequestError:
            return False

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def execute(code: str, base_url: str = "http://coderunner.local:8222") -> str:
    """
    Execute Python code and return stdout.

    Convenience function for one-off execution.

    Args:
        code: Python code to execute.
        base_url: Server URL. Defaults to http://coderunner.local:8222

    Returns:
        stdout from execution.

    Raises:
        RuntimeError: If execution produces stderr.

    Example:
        >>> from coderunner import execute
        >>> execute("print(2 + 2)")
        '4\\n'
    """
    with CodeRunner(base_url) as cr:
        result = cr.execute(code)
        if result.stderr:
            raise RuntimeError(result.stderr)
        return result.stdout
