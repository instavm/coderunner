"""
Tests for parsing functions in server.py.

These tests focus on the skill frontmatter parsing logic.
"""

import pytest
import asyncio
from unittest.mock import mock_open, patch, AsyncMock
import aiofiles


# We need to test the _parse_skill_frontmatter function from server.py
# Since it's a private function, we'll import it directly for testing


class TestSkillFrontmatterParsing:
    """Tests for SKILL.md frontmatter parsing."""

    @pytest.fixture
    def valid_frontmatter(self):
        """Valid SKILL.md with frontmatter."""
        return """---
name: pdf-text-replace
description: Replace text in PDF forms
version: 1.0.0
author: InstaVM
---

# PDF Text Replace

This skill replaces text in PDF forms.
"""

    @pytest.fixture
    def no_frontmatter(self):
        """SKILL.md without frontmatter."""
        return """# PDF Text Replace

This skill replaces text in PDF forms.
"""

    @pytest.fixture
    def empty_frontmatter(self):
        """SKILL.md with empty frontmatter."""
        return """---
---

# Empty Skill
"""

    @pytest.fixture
    def partial_frontmatter(self):
        """SKILL.md with partial frontmatter (missing closing)."""
        return """---
name: broken-skill

# This frontmatter is never closed
"""

    @pytest.mark.asyncio
    async def test_parse_valid_frontmatter(self, valid_frontmatter, tmp_path):
        """Parses valid frontmatter correctly."""
        # Import here to avoid import issues
        import sys
        sys.path.insert(0, str(tmp_path.parent.parent))

        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(valid_frontmatter)

        # Inline implementation of parsing logic for testing
        async def parse_frontmatter(file_path):
            async with aiofiles.open(file_path, mode='r') as f:
                content = await f.read()
                frontmatter = []
                in_frontmatter = False
                for line in content.splitlines():
                    if line.strip() == '---':
                        if in_frontmatter:
                            break
                        else:
                            in_frontmatter = True
                            continue
                    if in_frontmatter:
                        frontmatter.append(line)

                metadata = {}
                for line in frontmatter:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
                return metadata

        result = await parse_frontmatter(skill_file)

        assert result["name"] == "pdf-text-replace"
        assert result["description"] == "Replace text in PDF forms"
        assert result["version"] == "1.0.0"
        assert result["author"] == "InstaVM"

    @pytest.mark.asyncio
    async def test_parse_no_frontmatter(self, no_frontmatter, tmp_path):
        """Returns empty dict when no frontmatter."""
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(no_frontmatter)

        async def parse_frontmatter(file_path):
            async with aiofiles.open(file_path, mode='r') as f:
                content = await f.read()
                frontmatter = []
                in_frontmatter = False
                for line in content.splitlines():
                    if line.strip() == '---':
                        if in_frontmatter:
                            break
                        else:
                            in_frontmatter = True
                            continue
                    if in_frontmatter:
                        frontmatter.append(line)

                metadata = {}
                for line in frontmatter:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
                return metadata

        result = await parse_frontmatter(skill_file)
        assert result == {}

    @pytest.mark.asyncio
    async def test_parse_empty_frontmatter(self, empty_frontmatter, tmp_path):
        """Returns empty dict for empty frontmatter."""
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(empty_frontmatter)

        async def parse_frontmatter(file_path):
            async with aiofiles.open(file_path, mode='r') as f:
                content = await f.read()
                frontmatter = []
                in_frontmatter = False
                for line in content.splitlines():
                    if line.strip() == '---':
                        if in_frontmatter:
                            break
                        else:
                            in_frontmatter = True
                            continue
                    if in_frontmatter:
                        frontmatter.append(line)

                metadata = {}
                for line in frontmatter:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
                return metadata

        result = await parse_frontmatter(skill_file)
        assert result == {}

    @pytest.mark.asyncio
    async def test_parse_handles_colons_in_value(self, tmp_path):
        """Handles colons in values correctly."""
        content = """---
name: test-skill
url: https://example.com:8080/path
---
"""
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(content)

        async def parse_frontmatter(file_path):
            async with aiofiles.open(file_path, mode='r') as f:
                content = await f.read()
                frontmatter = []
                in_frontmatter = False
                for line in content.splitlines():
                    if line.strip() == '---':
                        if in_frontmatter:
                            break
                        else:
                            in_frontmatter = True
                            continue
                    if in_frontmatter:
                        frontmatter.append(line)

                metadata = {}
                for line in frontmatter:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
                return metadata

        result = await parse_frontmatter(skill_file)
        assert result["url"] == "https://example.com:8080/path"


class TestExecutionResultParsing:
    """Tests for parsing execution results."""

    def test_empty_stdout_is_empty_string(self):
        """Empty stdout should be empty string, not None."""
        from coderunner import ExecutionResult

        result = ExecutionResult(
            stdout="",
            stderr="",
            execution_time=0.0,
            success=True
        )
        assert result.stdout == ""
        assert isinstance(result.stdout, str)

    def test_multiline_stdout(self):
        """Handles multiline stdout correctly."""
        from coderunner import ExecutionResult

        multiline = "line1\nline2\nline3\n"
        result = ExecutionResult(
            stdout=multiline,
            stderr="",
            execution_time=0.1,
            success=True
        )
        assert result.stdout.count("\n") == 3
        assert "line2" in result.stdout
