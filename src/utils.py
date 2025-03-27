import os
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a JSON file
    
    Args:
        config_path: Path to the configuration file. If None, uses the default config.
        
    Returns:
        The configuration as a dictionary
    """
    if config_path is None:
        # Use default config path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        config_path = os.path.join(project_root, "config", "default_config.json")
    
    with open(config_path, "r") as f:
        return json.load(f)

def setup_environment() -> None:
    """Load environment variables from .env file"""
    # Load from .env file
    load_dotenv()
    
    # Validate required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}. "
            f"Please set them in your .env file or in your environment."
        )

def get_api_key() -> str:
    """Get the OpenAI API key from environment variables
    
    Returns:
        The API key
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it in your .env file or in your environment."
        )
    return api_key