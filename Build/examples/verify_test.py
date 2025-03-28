"""
Basic Verification Tool Test

This is a simplified test of the verification tool functionality
using the LLM as a Judge pattern.
"""
import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import libraries
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, trace

# Set OpenAI API key explicitly in the script
# This ensures it's available for both runner and tracing
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# Check for API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ ERROR: OPENAI_API_KEY not found!")
    print("Run 'python test_api_key.py' for instructions on setting up your API key.")
    sys.exit(1)

# Define evaluation feedback model
from pydantic import BaseModel
from typing import Literal

class EvaluationFeedback(BaseModel):
    """Feedback from the verifier agent"""
    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]

# Define a callable verification function
async def _verify_content_impl(content: str) -> str:
    """
    Implementation of the verification function.
    """
    print(f"Verifying content: {content[:50]}...")
    
    # Create the verifier agent
    verifier_agent = Agent(
        name="verifier_agent",
        instructions="""
        You are a verification agent that evaluates content quality.
        Your task is to critically assess content and provide specific, actionable feedback.
        
        Score content as:
        - "pass" if it's high quality with no significant issues
        - "needs_improvement" if it has minor issues that could be fixed
        - "fail" if it has major issues requiring complete revision
        
        Provide clear, specific feedback explaining your rating.
        """,
        output_type=EvaluationFeedback
    )
    
    # Run the verifier to evaluate content
    verifier_result = await Runner.run(
        verifier_agent,
        f"Please evaluate this content: {content}"
    )
    
    # Get the evaluation
    evaluation = verifier_result.final_output
    
    # Process based on evaluation
    if evaluation.score == "pass":
        return f"✅ PASSED VERIFICATION: {content}"
    else:
        # Create improvement agent
        improver_agent = Agent(
            name="improver_agent",
            instructions="You improve content based on specific feedback."
        )
        
        # Run improver
        improved_result = await Runner.run(
            improver_agent,
            f"""
            Original content: {content}
            
            Feedback: {evaluation.feedback}
            
            Please improve this content based on the feedback.
            """
        )
        
        return f"""
        🔄 IMPROVED CONTENT:
        {improved_result.final_output}
        
        Original feedback: {evaluation.feedback}
        Original score: {evaluation.score}
        """

async def main():
    print("\nBasic Verification Tool Test")
    print("==========================\n")
    
    # Create the function tool from our implementation
    verify_content = function_tool(_verify_content_impl)
    
    # Create a test agent with the verification tool
    agent = Agent(
        name="test_agent",
        instructions="""
        You are a test agent. When asked to verify content, use the verify_content tool.
        First generate the content requested, and then verify it with the tool.
        """,
        tools=[verify_content]
    )
    
    # Create a trace for the test
    with trace("Verification Tool Test"):
        # Run the agent
        result = await Runner.run(
            agent,
            "Generate a short paragraph about climate change and then verify it with the verify_content tool."
        )
        
        print("\nAgent Result:")
        print(result.final_output)
        
        # Now test the verification function directly
        direct_result = await _verify_content_impl("Climate change is a pressing global issue requiring immediate action.")
        
        print("\nDirect Verification Result:")
        print(direct_result)
        
        print("\nVerification test complete! Check the trace logs to see the verification spans.")

if __name__ == "__main__":
    asyncio.run(main())