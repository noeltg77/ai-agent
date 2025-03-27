"""
Simplified verification tool test
"""
import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents import Agent, Runner, function_tool, trace
from src.prompt_loader import PromptLoader
from src.verification import EvaluationFeedback

# Initialize the prompt loader
PromptLoader.initialize()

# Load verifier prompt
prompt_loader = PromptLoader()
verifier_prompt = prompt_loader.get_prompt("verifier_agent")

@function_tool
async def verify_content(content: str, content_type: str = "general") -> str:
    """
    Verify and improve content using the LLM as a Judge pattern.
    
    Args:
        content: The content to verify and improve
        content_type: The type of content (general, social_media, or image)
        
    Returns:
        The verified and improved content
    """
    print(f"Verify tool called for content: {content[:30]}...")
    
    # Create a verifier agent
    verifier_agent = Agent(
        name=f"verifier_agent",
        instructions=verifier_prompt,
        output_type=EvaluationFeedback,
    )
    
    # Run the verifier to evaluate the content
    verifier_result = await Runner.run(
        verifier_agent,
        f"Content to verify: {content}\n\nPlease evaluate this content."
    )
    
    # Get the evaluation
    evaluation = verifier_result.final_output
    
    # If content passes verification, return it
    if evaluation.score == "pass":
        return f"✅ VERIFIED: {content}"
    
    # If content needs improvement, create a new agent to improve it
    improver_agent = Agent(
        name="content_improver",
        instructions=f"You improve content based on feedback. Content type: {content_type}."
    )
    
    # Run the improver agent
    improved_result = await Runner.run(
        improver_agent,
        f"Original content: {content}\n\nFeedback: {evaluation.feedback}\n\nPlease improve this content based on the feedback."
    )
    
    # Return improved content
    return f"🔄 IMPROVED: {improved_result.final_output}\n\nOriginal feedback: {evaluation.feedback}"

async def main():
    print("Verification Tool Test")
    print("=====================")
    
    # Create a test agent with the verification tool
    test_agent = Agent(
        name="test_agent",
        instructions="You are a test agent. When asked to verify content, use the verify_content tool.",
        tools=[verify_content]
    )
    
    # Create a trace for the entire test
    with trace("Verification Tool Test"):
        # Run the agent with a request to verify content
        result = await Runner.run(
            test_agent,
            "Generate a short paragraph about climate change and then verify it using the verify_content tool."
        )
        
        print("\nAgent Result:")
        print(result.final_output)
        
        # Now try using the verify tool directly
        direct_result = await verify_content("Climate change is a global crisis requiring immediate action.")
        
        print("\nDirect Verification Result:")
        print(direct_result)

if __name__ == "__main__":
    asyncio.run(main())