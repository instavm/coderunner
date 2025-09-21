"""Tests for language processor functionality"""

import pytest
from core.language_processor import LanguageProcessor


class TestLanguageProcessor:
    """Test language processing functionality"""
    
    def test_normalize_language(self):
        """Test language normalization"""
        # Direct matches
        assert LanguageProcessor.normalize_language("python") == "python"
        assert LanguageProcessor.normalize_language("bash") == "bash"
        assert LanguageProcessor.normalize_language("javascript") == "javascript"
        
        # Aliases
        assert LanguageProcessor.normalize_language("py") == "python"
        assert LanguageProcessor.normalize_language("python3") == "python"
        assert LanguageProcessor.normalize_language("sh") == "bash"
        assert LanguageProcessor.normalize_language("shell") == "bash"
        assert LanguageProcessor.normalize_language("js") == "javascript"
        assert LanguageProcessor.normalize_language("node") == "javascript"
        
        # Case insensitive
        assert LanguageProcessor.normalize_language("PYTHON") == "python"
        assert LanguageProcessor.normalize_language("Python") == "python"
        assert LanguageProcessor.normalize_language("BASH") == "bash"
        
        # Unknown languages default to python
        assert LanguageProcessor.normalize_language("unknown") == "python"
        assert LanguageProcessor.normalize_language("") == "python"
        assert LanguageProcessor.normalize_language(None) == "python"
        
    def test_validate_language(self):
        """Test language validation"""
        # Valid languages
        assert LanguageProcessor.validate_language("python") == True
        assert LanguageProcessor.validate_language("bash") == True
        assert LanguageProcessor.validate_language("javascript") == True
        assert LanguageProcessor.validate_language("py") == True
        assert LanguageProcessor.validate_language("sh") == True
        
        # Unknown languages are still valid (normalized to python)
        assert LanguageProcessor.validate_language("unknown") == True
        assert LanguageProcessor.validate_language("") == True
        
    def test_detect_language(self):
        """Test automatic language detection"""
        # Python by default
        assert LanguageProcessor.detect_language("print('hello')") == "python"
        assert LanguageProcessor.detect_language("x = 5") == "python"
        
        # Bash detection
        assert LanguageProcessor.detect_language("%%bash\\nls -la") == "bash"
        assert LanguageProcessor.detect_language("!ls -la") == "bash"
        assert LanguageProcessor.detect_language("ls -la") == "bash"
        assert LanguageProcessor.detect_language("mkdir test") == "bash"
        assert LanguageProcessor.detect_language("cat file.txt") == "bash"
        
        # JavaScript detection
        assert LanguageProcessor.detect_language("%%javascript\\nconsole.log('hi')") == "javascript"
        assert LanguageProcessor.detect_language("npm install express") == "javascript"
        assert LanguageProcessor.detect_language("node app.js") == "javascript"
        assert LanguageProcessor.detect_language("yarn add react") == "javascript"
        
    def test_preprocess_command(self):
        """Test command preprocessing"""
        # Python commands remain unchanged
        python_code = "print('hello world')"
        assert LanguageProcessor.preprocess_command(python_code, "python") == python_code
        
        # Bash commands get %%bash prefix if needed
        bash_cmd = "ls -la"
        expected = "%%bash\\nls -la"
        assert LanguageProcessor.preprocess_command(bash_cmd, "bash") == expected
        
        # Already prefixed bash commands remain unchanged
        prefixed_bash = "%%bash\\nls -la"
        assert LanguageProcessor.preprocess_command(prefixed_bash, "bash") == prefixed_bash
        
        # JavaScript commands get %%javascript prefix
        js_code = "console.log('hello')"
        expected = "%%javascript\\nconsole.log('hello')"
        assert LanguageProcessor.preprocess_command(js_code, "javascript") == expected
        
        # ! commands get converted to %%bash
        bang_cmd = "!pip install requests"
        expected = "%%bash\\npip install requests"
        assert LanguageProcessor.preprocess_command(bang_cmd, "python") == expected
        
    def test_get_supported_languages(self):
        """Test getting supported languages"""
        languages = LanguageProcessor.get_supported_languages()
        
        # Check required languages are present
        assert "python" in languages
        assert "bash" in languages
        assert "javascript" in languages
        
        # Check structure
        assert "aliases" in languages["python"]
        assert "magic" in languages["python"]
        
        # Check aliases
        assert "py" in languages["python"]["aliases"]
        assert "sh" in languages["bash"]["aliases"]
        assert "js" in languages["javascript"]["aliases"]