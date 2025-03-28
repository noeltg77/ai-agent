"""
Simplified Verification Test
"""
import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Set OpenAI API key explicitly before importing any OpenAI-related packages
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    print(f"API Key set: {api_key[:5]}...{api_key[-4:]}")
else:
    print("No OpenAI API key found! Please add it to .env")
    sys.exit(1)

# Add the project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import OpenAI and agents modules
from pydantic import BaseModel
from typing import Literal
from agents import Agent, Runner, trace

# Define our feedback model
class Feedback(BaseModel):
    message: str
    rating: Literal["good", "needs_work", "poor"]

# Create an extremely simple verification system
async def verify_text(text: str) -> str:
    """Verify and improve text"""
    
    # Create a verifier agent
    verifier = Agent(
        name="simple_verifier", 
        instructions="You evaluate content quality and provide feedback.",
        output_type=Feedback
    )
    
    # Evaluate the text
    verification = await Runner.run(
        verifier,
        f"Evaluate this text: {text}"
    )
    
    # Extract the feedback
    feedback = verification.final_output
    
    if feedback.rating == "good":
        return f"✅ Verified: {text}"
    else:
        # Create an improver agent
        improver = Agent(
            name="simple_improver",
            instructions="You improve content based on feedback."
        )
        
        # Get improved version
        improved = await Runner.run(
            improver,
            f"Original: {text}\nFeedback: {feedback.message}\nCreate an improved version."
        )
        
        return f"🔄 Improved: {improved.final_output}"

# Define our verification tool
def get_verify_tool():
    """Return a dictionary with tool definition"""
    return {
        "type": "function",
        "function": {
            "name": "verify_content",
            "description": "Verify and improve content quality",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to verify"
                    }
                },
                "required": ["content"]
            }
        }
    }

async def main():
    print("\nSimple Verification Test")
    print("=======================\n")
    
    # Test direct verification first
    test_text = "Climate change is bad for the environment."
    print(f"Testing direct verification: {test_text}")
    
    # Use with trace to ensure proper tracing
    with trace("Direct Verification Test"):
        result = await verify_text(test_text)
        print(f"\nDirect result: {result}\n")
    
    # Now test agent with tool
    agent = Agent(
        name="test_agent",
        instructions="""
        You help users by creating and verifying content.
        When asked to verify content, use the verify_content tool.
        """,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "verify_content",
                    "description": "Verify and improve content quality",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The content to verify"
                            }
                        },
                        "required": ["content"]
                    }
                }
            }
        ]
    )
    
    # Register tool handler for the agent
    async def handle_verify_content(params):
        content = params.get("content", "")
        return await verify_text(content)
    
    agent.register_tool_handler("verify_content", handle_verify_content)
    
    # Test the agent with the tool
    with trace("Agent Verification Test"):
        agent_result = await Runner.run(
            agent,
            "Write a short sentence about renewable energy and verify it."
        )
        
        print(f"Agent result: {agent_result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())