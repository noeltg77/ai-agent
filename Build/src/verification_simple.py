"""
Simplified verification module for testing purposes
"""
from __future__ import annotations

from typing import Literal

from agents import Agent, Runner, function_tool, trace
from pydantic import BaseModel

from src.prompt_loader import PromptLoader

# Initialize prompt loader
PromptLoader.initialize()

# Load verifier prompt
verifier_prompt = PromptLoader.get_prompt("verifier_agent")

class EvaluationFeedback(BaseModel):
    """
    Output model for the verifier agent.
    """
    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]

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
        name="verifier_agent",
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

# The Verification class below is the more complex implementation for integration
# with the rest of the system, but it's not needed for simple testing
class Verification:
    """
    Implements the "LLM as a Judge" pattern for verifying and improving agent outputs.
    """
    
    def __init__(self, max_attempts: int = 3):
        """
        Initialize the verification system.
        
        Args:
            max_attempts: Maximum number of improvement attempts before returning the best result
        """
        # Load the verifier agent prompt
        self.verifier_prompt = verifier_prompt
        self.max_attempts = max_attempts
    
    def get_verification_tool(self):
        """
        Get the verification function tool.
        """
        return verify_content