"""
Simple test for verification tool
"""
import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents import Agent, Runner, function_tool
from src.prompt_loader import PromptLoader
from agents.tracing import get_current_span

# Initialize the prompt loader
PromptLoader.initialize()

@function_tool
def sample_tool(message: str) -> str:
    """A simple tool that echoes back a message."""
    print(f"Sample tool called with: {message}")
    
    # Print current span info
    span = get_current_span()
    if span:
        print(f"Current span in sample_tool: {span.name}")
    else:
        print("No current span in sample_tool")
        
    return f"Echo: {message}"

async def main():
    print("Testing function tools")
    print("=====================")
    
    # Create a test agent with both the sample tool
    agent = Agent(
        name="tool_test_agent",
        instructions="You are a test agent that will use tools when asked.",
        tools=[sample_tool]
    )
    
    # Run the agent
    result = await Runner.run(
        agent, 
        "Please use the sample_tool with the message 'This is a test'"
    )
    
    print("\nAgent result:")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())