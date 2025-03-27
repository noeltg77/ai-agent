"""
Prompt Loader - Utility for loading prompts from files
"""
import os
from pathlib import Path
from typing import Dict, Optional

class PromptLoader:
    """
    A utility class to load prompts from files in the prompts directory.
    """
    _prompts_cache: Dict[str, str] = {}
    _prompts_dir: str = None
    
    @classmethod
    def initialize(cls, prompts_dir: Optional[str] = None) -> None:
        """
        Initialize the prompt loader with the given prompts directory.
        If not provided, defaults to the 'prompts' directory in the project root.
        
        Args:
            prompts_dir: Optional path to the prompts directory
        """
        if prompts_dir:
            cls._prompts_dir = prompts_dir
        else:
            # Default to the 'prompts' directory in the project root
            project_root = Path(__file__).parent.parent
            cls._prompts_dir = os.path.join(project_root, 'prompts')
        
        # Clear the cache when initializing
        cls._prompts_cache.clear()
    
    @classmethod
    def get_prompt(cls, prompt_name: str) -> str:
        """
        Get a prompt by name. Prompts are loaded from files named <prompt_name>.txt
        in the prompts directory. Prompts are cached after first load.
        
        Args:
            prompt_name: The name of the prompt file (without .txt extension)
            
        Returns:
            The prompt text
            
        Raises:
            FileNotFoundError: If the prompt file doesn't exist
        """
        # Initialize with default if not done yet
        if cls._prompts_dir is None:
            cls.initialize()
            
        # Check if the prompt is already cached
        if prompt_name in cls._prompts_cache:
            return cls._prompts_cache[prompt_name]
        
        # Construct the prompt file path
        prompt_file = os.path.join(cls._prompts_dir, f"{prompt_name}.txt")
        
        # Load the prompt from the file
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read()
        
        # Cache the prompt
        cls._prompts_cache[prompt_name] = prompt
        
        return prompt
    
    @classmethod
    def reload_prompt(cls, prompt_name: str) -> str:
        """
        Force reload a prompt from its file, bypassing the cache.
        
        Args:
            prompt_name: The name of the prompt file (without .txt extension)
            
        Returns:
            The reloaded prompt text
        """
        # Remove from cache if present
        if prompt_name in cls._prompts_cache:
            del cls._prompts_cache[prompt_name]
        
        # Load and return the prompt
        return cls.get_prompt(prompt_name)
    
    @classmethod
    def reload_all(cls) -> None:
        """
        Force reload all prompts from their files, clearing the cache.
        """
        cls._prompts_cache.clear()