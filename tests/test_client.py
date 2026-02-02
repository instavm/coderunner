"""
Tests for the CodeRunner Python client library.

These tests mock HTTP responses to avoid requiring a running server.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import httpx

from coderunner import CodeRunner, ExecutionResult, BrowserResult, execute


class TestCodeRunnerClient:
    """Tests for the CodeRunner client class."""

    def test_init_default_url(self):
        """Client uses default URL when none provided."""
        with patch("httpx.Client"):
            cr = CodeRunner()
            assert cr.base_url == "http://coderunner.local:8222"

    def test_init_custom_url(self):
        """Client uses custom URL when provided."""
        with patch("httpx.Client"):
            cr = CodeRunner(base_url="http://localhost:9000")
            assert cr.base_url == "http://localhost:9000"

    def test_init_strips_trailing_slash(self):
        """Client strips trailing slash from URL."""
        with patch("httpx.Client"):
            cr = CodeRunner(base_url="http://localhost:9000/")
            assert cr.base_url == "http://localhost:9000"

    def test_execute_success(self, execute_success_response):
        """Execute returns correct result on success."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = execute_success_response
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            cr = CodeRunner()
            result = cr.execute("print(2 + 2)")

            assert isinstance(result, ExecutionResult)
            assert result.stdout == "4\n"
            assert result.stderr == ""
            assert result.execution_time == 0.05
            assert result.success is True
            mock_client.post.assert_called_once()

    def test_execute_with_error(self, execute_error_response):
        """Execute returns error in stderr when code fails."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = execute_error_response
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            cr = CodeRunner()
            result = cr.execute("print(undefined)")

            assert result.success is False
            assert "NameError" in result.stderr

    def test_execute_calls_correct_endpoint(self):
        """Execute calls /v1/execute endpoint."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"stdout": "", "stderr": "", "execution_time": 0}
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            cr = CodeRunner(base_url="http://test:8222")
            cr.execute("pass")

            mock_client.post.assert_called_with(
                "http://test:8222/v1/execute",
                json={"code": "pass"}
            )

    def test_browse_success(self, browse_success_response):
        """Browse returns content on success."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = browse_success_response
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            cr = CodeRunner()
            result = cr.browse("https://example.com")

            assert isinstance(result, BrowserResult)
            assert "Example Domain" in result.content
            assert result.url == "https://example.com"
            assert result.success is True

    def test_browse_error(self, browse_error_response):
        """Browse returns error on failure."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = browse_error_response
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            cr = CodeRunner()
            result = cr.browse("https://invalid.local")

            assert result.success is False
            assert result.error == "Connection refused"

    def test_health_check_healthy(self):
        """Health returns True when server is healthy."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            cr = CodeRunner()
            assert cr.health() is True

    def test_health_check_unhealthy(self):
        """Health returns False when server is down."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client_class.return_value = mock_client

            cr = CodeRunner()
            assert cr.health() is False

    def test_health_check_server_error(self):
        """Health returns False on server error."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            cr = CodeRunner()
            assert cr.health() is False

    def test_context_manager(self):
        """Client works as context manager."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            with CodeRunner() as cr:
                assert cr is not None

            mock_client.close.assert_called_once()

    def test_close_method(self):
        """Close method closes underlying client."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            cr = CodeRunner()
            cr.close()

            mock_client.close.assert_called_once()


class TestExecuteFunction:
    """Tests for the execute() convenience function."""

    def test_execute_function_returns_stdout(self):
        """Execute function returns stdout on success."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "stdout": "hello\n",
                "stderr": "",
                "execution_time": 0.01
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = execute("print('hello')")
            assert result == "hello\n"

    def test_execute_function_raises_on_error(self):
        """Execute function raises RuntimeError on stderr."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "stdout": "",
                "stderr": "SyntaxError: invalid syntax",
                "execution_time": 0.01
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            with pytest.raises(RuntimeError) as exc_info:
                execute("invalid python code")

            assert "SyntaxError" in str(exc_info.value)

    def test_execute_function_custom_url(self):
        """Execute function accepts custom base URL."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "stdout": "ok",
                "stderr": "",
                "execution_time": 0.01
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            execute("pass", base_url="http://custom:9000")

            mock_client.post.assert_called_with(
                "http://custom:9000/v1/execute",
                json={"code": "pass"}
            )


class TestDataClasses:
    """Tests for data classes."""

    def test_execution_result_fields(self):
        """ExecutionResult has correct fields."""
        result = ExecutionResult(
            stdout="output",
            stderr="error",
            execution_time=1.5,
            success=False
        )
        assert result.stdout == "output"
        assert result.stderr == "error"
        assert result.execution_time == 1.5
        assert result.success is False

    def test_browser_result_fields(self):
        """BrowserResult has correct fields."""
        result = BrowserResult(
            content="page content",
            url="https://example.com",
            success=True,
            error=None
        )
        assert result.content == "page content"
        assert result.url == "https://example.com"
        assert result.success is True
        assert result.error is None

    def test_browser_result_with_error(self):
        """BrowserResult can hold error."""
        result = BrowserResult(
            content="",
            url="https://invalid.local",
            success=False,
            error="Connection refused"
        )
        assert result.success is False
        assert result.error == "Connection refused"
