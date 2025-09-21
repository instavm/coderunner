"""
Cloud execution via InstaVM - same interface as local CodeRunner

This module provides the cloud migration path for CodeRunner users.
Simply change your import to use cloud execution with the same interface.

Example migration:
    # Before (local)
    from coderunner import CodeRunner
    runner = CodeRunner()
    
    # After (cloud) - just add API key
    from coderunner.cloud import InstaVM as CodeRunner  
    runner = CodeRunner(api_key="your-key")
    
All methods have the same interface between local and cloud execution.
"""

try:
    from instavm import InstaVM as _InstaVM
    
    # Re-export with enhanced docstring for migration
    class InstaVM(_InstaVM):
        """
        Cloud execution client (InstaVM) with CodeRunner-compatible interface.
        
        This is an alias to the InstaVM class for easy migration from local CodeRunner.
        All methods have the same interface as local CodeRunner.
        
        Usage:
            from coderunner.cloud import InstaVM as CodeRunner
            runner = CodeRunner(api_key="your-api-key")
            result = runner.execute("print('Hello Cloud!')")
            
        Migration from local CodeRunner:
            1. Change import: from coderunner.cloud import InstaVM as CodeRunner
            2. Add API key: runner = CodeRunner(api_key="your-key")
            3. Everything else stays the same!
        """
        
        def __init__(self, api_key: str, base_url: str = "https://api.instavm.io", **kwargs):
            """
            Initialize cloud execution client
            
            Args:
                api_key: InstaVM API key (required)
                base_url: API base URL (default: https://api.instavm.io)
                **kwargs: Additional arguments passed to InstaVM
            """
            if not api_key:
                raise ValueError(
                    "API key is required for cloud execution.\n"
                    "Get your API key from: https://instavm.io\n\n"
                    "Or use local execution:\n"
                    "  from coderunner import CodeRunner\n"
                    "  runner = CodeRunner()  # No API key needed"
                )
            
            super().__init__(api_key=api_key, base_url=base_url, **kwargs)
            
    # Also export the original class for direct access
    __all__ = ["InstaVM"]
    
except ImportError:
    # InstaVM package not installed
    class InstaVM:
        """Placeholder InstaVM class when package not installed"""
        
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "InstaVM package not found. Install with:\n"
                "  pip install instavm\n\n"
                "Or use local CodeRunner:\n"
                "  from coderunner import CodeRunner\n"
                "  runner = CodeRunner()  # No API key needed"
            )
    
    __all__ = ["InstaVM"]