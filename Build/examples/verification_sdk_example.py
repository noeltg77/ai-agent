"""
Verification Example using the SDK-style implementation
"""
import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set OpenAI API key explicitly before importing any OpenAI-related packages
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    print(f"API Key set: {api_key[:5]}...{api_key[-4:]}")
else:
    print("No OpenAI API key found! Please add it to .env")
    sys.exit(1)

from agents import Agent, Runner, trace, gen_trace_id, function_tool
from src.verification_sdk import Verification, EvaluationFeedback
from src.prompt_loader import PromptLoader

# Initialize the prompt loader
PromptLoader.initialize()

async def run_verification_demo():
    """
    Demo the SDK-style verification capabilities
    """
    # Generate a trace ID for the demo
    main_trace_id = gen_trace_id()
    
    print("\n===== SDK-STYLE VERIFICATION DEMO =====\n")
    print(f"Main Trace ID: {main_trace_id}")
    print("This demo shows how content verification improves agent outputs through multiple iterations.")
    print("All verification activities will be visible in the trace logs under this trace ID.\n")
    
    # Create a trace for the entire demo
    with trace("SDK Verification Demo", trace_id=main_trace_id):
        # Create the verification system
        verification = Verification(max_attempts=3)
        
        # Create test agents
        content_agent = Agent(
            name="content_generator",
            instructions="You generate clear, concise content based on user requests."
        )
        
        print("\n----- DIRECT VERIFICATION PROCESS -----\n")
        
        # Test the direct verification process
        query = input("Enter a topic for content generation (or press Enter for default): ")
        if not query:
            query = "Climate change impacts on agriculture"
            print(f"Using default topic: {query}")
        
        print("\nGenerating and verifying content...\n")
        
        # Run the verification process
        final_content, verification_history = await verification.verify_and_improve(
            content_agent,
            f"Write a paragraph about {query}",
            trace_name="Direct verification example"
        )
        
        # Print the verification process
        print("Verification process:\n")
        for i, attempt in enumerate(verification_history):
            print(f"Attempt {i+1}:")
            print(f"Score: {attempt.get('score', 'N/A')}")
            print(f"Feedback: {attempt.get('feedback', 'N/A')}")
            print("-" * 50)
        
        print("\nFinal verified content:")
        print(final_content)
        
        print("\n----- TOOL-BASED VERIFICATION -----\n")
        
        # Create a test agent with verification tool
        tool_agent = Agent(
            name="tool_using_agent",
            instructions="""
            You are an agent that creates content and verifies it.
            When asked to verify content, always use the verify_content tool.
            """,
            tools=[verification.get_verify_function_tool()]
        )
        
        # Run the test agent
        tool_query = input("Enter a topic for tool-based verification (or press Enter for default): ")
        if not tool_query:
            tool_query = "Renewable energy technologies"
            print(f"Using default topic: {tool_query}")
        
        print("\nTesting tool-based verification...\n")
        
        tool_result = await Runner.run(
            tool_agent,
            f"Write a paragraph about {tool_query} and then use the verify_content tool to verify it."
        )
        
        print("\nTool-based verification result:")
        print(tool_result.final_output)

async def main():
    print("SDK-Style Verification Demo")
    print("===========================")
    print("This demonstration shows verification using the OpenAI Agents SDK pattern")
    print("This approach ensures correct tracing and tool registration.")
    
    await run_verification_demo()
    
    print("\nThank you for trying the SDK-Style Verification Demo!")
    print("Check the trace logs to see all verification activities.")

if __name__ == "__main__":
    asyncio.run(main())