"""
Verification Tool Test - A focused test for the verification function tool
"""
import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents import Agent, Runner, trace, function_tool
from agents.tracing import custom_span, Trace
from src.prompt_loader import PromptLoader
from src.verification import Verification

# Initialize the prompt loader
PromptLoader.initialize()

@function_tool
def sample_function_tool(message: str) -> str:
    """
    A sample function tool to test against
    
    Args:
        message: A message to echo
        
    Returns:
        The message, echoed back
    """
    print(f"Sample tool called with: {message}")
    return f"Echo: {message}"

async def main():
    """
    Test the verification function tool in isolation
    """
    print("Verification Function Tool Test")
    print("=============================")
    
    # Create a trace for the entire test
    main_trace_id = "trace_verification_tool_test_" + os.urandom(4).hex()
    main_trace = trace(
        workflow_name="Verification Tool Test",
        trace_id=main_trace_id
    )
    
    try:
        main_trace.start(mark_as_current=True)
        
        # Create the verification tool
        verification = Verification(max_attempts=2)
        verify_tool = verification.get_verify_function_tool()
        
        # Print information about the tool
        print(f"Tool name: {getattr(verify_tool, 'name', 'unknown')}")
        print(f"Tool type: {type(verify_tool)}")
        
        # Create a test agent with multiple tools including the verification tool
        test_agent = Agent(
            name="test_agent_with_tools",
            instructions="""
            You are a test agent with multiple tools available. When asked to verify content, use the verify_content tool.
            When asked to echo a message, use the sample_function_tool.
            Always use tools when available instead of generating content directly.
            """,
            tools=[
                verify_tool,
                sample_function_tool
            ]
        )
        
        # Run the agent with a request specifically asking to use verification
        print("\nRunning the agent with explicit verification request...")
        result = await Runner.run(
            test_agent,
            "First, generate a short paragraph about climate change. Then, explicitly verify it using the verify_content tool."
        )
        
        print("\nAgent result:")
        print(result.final_output)
        
        # Now test the verify tool directly
        print("\nDirectly testing the verification tool...")
        direct_result = await verify_tool("Climate change is a significant global challenge requiring immediate action.", "general")
        
        print("\nDirect verification result:")
        print(direct_result)
        
        print(f"\nMain trace ID: {main_trace_id}")
        print("Check the trace logs for verification tool spans.")
    
    finally:
        main_trace.finish(reset_current=True)

if __name__ == "__main__":
    asyncio.run(main())