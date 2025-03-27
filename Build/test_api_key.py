"""
API Key Test Script

This script checks if your OpenAI API key is properly configured.
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_api_key():
    """Check if the OpenAI API key is set up correctly"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY not found!")
        print("\nTo fix this:")
        print("1. Copy .env.example to .env: cp .env.example .env")
        print("2. Edit .env and add your API key: OPENAI_API_KEY=your_key_here")
        return False
    
    if api_key == "your_openai_api_key_here":
        print("❌ OPENAI_API_KEY is set to the example value!")
        print("\nTo fix this:")
        print("1. Edit your .env file")
        print("2. Replace the example API key with your actual OpenAI API key")
        return False
    
    # Print masked API key for verification
    masked_key = f"{api_key[:5]}...{api_key[-4:]}"
    print(f"✅ OPENAI_API_KEY found: {masked_key}")
    print("\nYour API key is configured correctly.")
    return True

def main():
    """Main function to run the API key test"""
    print("\n=== OpenAI API Key Test ===\n")
    
    api_key_ok = check_api_key()
    
    # Check for other useful keys
    replicate_key = os.getenv("REPLICATE_API_TOKEN")
    if replicate_key:
        print("\n✅ REPLICATE_API_TOKEN found (needed for image generation)")
    else:
        print("\nℹ️ REPLICATE_API_TOKEN not found (only needed if using image generation)")
    
    # Print final summary
    print("\n=== Summary ===")
    if api_key_ok:
        print("Your environment is properly configured for basic agent functionality.")
        print("You can now run examples like:")
        print("  python cli.py multi-agent simple")
    else:
        print("Your API key configuration needs to be fixed before running examples.")
    
    print("\nFor more information, see the README.md file.")

if __name__ == "__main__":
    main()