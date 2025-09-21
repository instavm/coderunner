"""Language processing for multi-language execution support"""

from typing import Dict, Any
from .exceptions import LanguageError


class LanguageProcessor:
    """Handle multi-language execution like InstaVM"""
    
    SUPPORTED_LANGUAGES = {
        "python": {"aliases": ["py", "python3"], "magic": None},
        "bash": {"aliases": ["sh", "shell", "bash"], "magic": "%%bash"},
        "javascript": {"aliases": ["js", "node"], "magic": "%%javascript"},
    }
    
    @classmethod
    def preprocess_command(cls, command: str, language: str) -> str:
        """
        Preprocess command based on language
        
        Args:
            command: Raw command to execute
            language: Target language
            
        Returns:
            Processed command ready for execution
        """
        language = cls.normalize_language(language)
        
        if language == "bash":
            if not command.strip().startswith("%%bash"):
                return f"%%bash\n{command}"
        elif language == "javascript":
            if not command.strip().startswith("%%javascript"):
                return f"%%javascript\n{command}"
        elif command.startswith("!"):
            # Convert ! commands to %%bash 
            return f"%%bash\n{command[1:]}"
            
        return command
        
    @classmethod
    def detect_language(cls, command: str) -> str:
        """
        Auto-detect language from command
        
        Args:
            command: Command to analyze
            
        Returns:
            Detected language name
        """
        command = command.strip()
        
        if command.startswith("%%bash"):
            return "bash"
        elif command.startswith("%%javascript"):
            return "javascript"
        elif command.startswith("!"):
            return "bash"
        elif any(cmd in command.lower() for cmd in ["npm", "node", "yarn"]):
            return "javascript"
        elif any(cmd in command.lower() for cmd in ["ls", "cd", "mkdir", "cat", "grep"]):
            return "bash"
        else:
            return "python"
            
    @classmethod
    def normalize_language(cls, language: str) -> str:
        """
        Normalize language name to standard form
        
        Args:
            language: Raw language name
            
        Returns:
            Normalized language name
        """
        if not language:
            return "python"
            
        language = language.lower().strip()
        
        # Check direct match
        if language in cls.SUPPORTED_LANGUAGES:
            return language
            
        # Check aliases
        for lang, config in cls.SUPPORTED_LANGUAGES.items():
            if language in config["aliases"]:
                return lang
                
        # Default to Python for unknown languages
        return "python"
        
    @classmethod
    def validate_language(cls, language: str) -> bool:
        """
        Validate if language is supported
        
        Args:
            language: Language to validate
            
        Returns:
            True if supported, False otherwise
        """
        normalized = cls.normalize_language(language)
        return normalized in cls.SUPPORTED_LANGUAGES
        
    @classmethod
    def get_supported_languages(cls) -> Dict[str, Any]:
        """
        Get list of supported languages with their configurations
        
        Returns:
            Dictionary of supported languages
        """
        return cls.SUPPORTED_LANGUAGES.copy()