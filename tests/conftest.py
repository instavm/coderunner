"""
Pytest configuration and fixtures for CodeRunner tests.
"""

import pytest
from unittest.mock import Mock, patch
import httpx


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.Client for testing without network calls."""
    with patch("httpx.Client") as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def execute_success_response():
    """Standard successful execution response."""
    return {
        "stdout": "4\n",
        "stderr": "",
        "execution_time": 0.05
    }


@pytest.fixture
def execute_error_response():
    """Execution response with error."""
    return {
        "stdout": "",
        "stderr": "NameError: name 'undefined' is not defined",
        "execution_time": 0.02
    }


@pytest.fixture
def browse_success_response():
    """Successful browser content response."""
    return {
        "readable_content": {
            "content": "Example Domain\nThis domain is for use in examples."
        },
        "status": "success"
    }


@pytest.fixture
def browse_error_response():
    """Browser error response."""
    return {
        "status": "error",
        "error": "Connection refused"
    }


@pytest.fixture
def health_success_response():
    """Healthy server response."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "healthy", "version": "0.1.0"}
    return mock_response


@pytest.fixture
def skill_frontmatter_content():
    """Sample SKILL.md content with frontmatter."""
    return """---
name: pdf-text-replace
description: Replace text in PDF forms
version: 1.0.0
---

# PDF Text Replace

This skill replaces text in PDF forms.
"""


@pytest.fixture
def skill_no_frontmatter_content():
    """Sample SKILL.md content without frontmatter."""
    return """# PDF Text Replace

This skill replaces text in PDF forms.
"""
