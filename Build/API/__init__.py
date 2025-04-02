"""
API module for the multi-agent system.
"""
# Import key components to make them available directly from the API package
import os
import sys

# Add current directory to path to enable relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    print(f"Added {current_dir} to Python path")

# Optional: Set version
__version__ = "1.0.0"