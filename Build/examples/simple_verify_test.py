"""
Ultra-simplified test for verification
"""
import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents import Agent, Runner, trace
from src.verification_simple import verify_content
from src.prompt_loader import PromptLoader
from src.utils import setup_environment, get_api_key

# Initialize environment and API key
try:
    setup_environment()
    
    # Print API key status
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"OpenAI API Key found: {api_key[:5]}...{api_key[-4:]}")
    else:
        print("WARNING: No OpenAI API Key found in environment!")
except Exception as e:
    print(f"⚠️ Environment setup error: {e}")
    print("You need to create a .env file with your OPENAI_API_KEY.")
    print("Copy .env.example to .env and add your API key.")

# Initialize the prompt loader
PromptLoader.initialize()

async def main():
    print("Simple Verification Tool Test")
    print("===========================")
    
    # Create a test agent with the verification tool
    test_agent = Agent(
        name="simple_test_agent",
        instructions="""
        You are a test agent with a verification tool.
        When asked to verify content, use the verify_content tool.
        Always use tools when asked to use them.
        """,
        tools=[verify_content]
    )
    
    # Create a trace for the entire test
    with trace("Simple Verification Test"):
        # Run the agent
        result = await Runner.run(
            test_agent,
            "Generate a one-sentence statement about climate change and then use the verify_content tool to verify it."
        )
        
        print("\nAgent Result:")
        print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())