import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Set OpenAI API key explicitly for proper tracing
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    print(f"API Key set: {api_key[:5]}...{api_key[-4:]}")
else:
    print("Warning: No OpenAI API key found in .env file")

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents import Agent, Runner, trace, ItemHelpers, WebSearchTool, function_tool
from src.prompt_loader import PromptLoader
from src.tools import generate_image
from src.verification_sdk import Verification

# Initialize the prompt loader
PromptLoader.initialize()

# Create the verification system
verification = Verification(max_attempts=2)

# Define our specialized agents
research_agent = Agent(
    name="research_agent",
    instructions=PromptLoader.get_prompt("research_agent"),
    tools=[WebSearchTool(user_location={"type": "approximate", "city": "London"})]
)

# Create platform-specific social media agents
linkedin_agent = Agent(
    name="linkedin_agent",
    instructions=PromptLoader.get_prompt("linkedin_agent"),
)

instagram_agent = Agent(
    name="instagram_agent",
    instructions=PromptLoader.get_prompt("instagram_agent"),
)

# Create the social media orchestrator agent
social_media_agent = Agent(
    name="social_media_agent",
    instructions=PromptLoader.get_prompt("social_media_agent"),
    tools=[
        linkedin_agent.as_tool(
            tool_name="create_linkedin_content",
            tool_description="Create professional content optimized for LinkedIn's business-focused audience.",
        ),
        instagram_agent.as_tool(
            tool_name="create_instagram_content",
            tool_description="Create visually-oriented content optimized for Instagram's lifestyle-focused platform.",
        ),
    ],
)

summarizer_agent = Agent(
    name="summarizer_agent",
    instructions=PromptLoader.get_prompt("summarizer_agent"),
)

# Create the graphic designer agent
graphic_designer_agent = Agent(
    name="graphic_designer_agent",
    instructions=PromptLoader.get_prompt("graphic_designer_agent"),
    tools=[generate_image],
)

# The main orchestrator agent that delegates tasks and combines results
orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=PromptLoader.get_prompt("orchestrator_agent"),
    tools=[
        research_agent.as_tool(
            tool_name="research",
            tool_description="Use this tool to search for factual information or answer knowledge-based questions.",
        ),
        social_media_agent.as_tool(
            tool_name="create_social_media_content",
            tool_description="Use this tool to generate optimized content for different social media platforms based on the target audience and goals.",
        ),
        graphic_designer_agent.as_tool(
            tool_name="create_image",
            tool_description="Use this tool to generate custom images based on detailed descriptions. Useful for creating visuals for social media posts.",
        ),
        summarizer_agent.as_tool(
            tool_name="summarize",
            tool_description="Use this tool to create concise summaries of research and social media content.",
        ),
        # Add the verification tool
        verification.get_verify_function_tool(),
    ],
)

async def main():
    print("Multi-Agent System Example with Verification")
    print("===========================================")
    print("This system uses multiple specialized agents coordinated by an orchestrator.")
    print("It includes content verification capabilities using the LLM as a Judge pattern.")
    print("You can:")
    print("  1. Ask research questions (uses web search)")
    print("  2. Request social media content for specific platforms:")
    print("     - LinkedIn (professional, business-focused)")
    print("     - Instagram (visual, lifestyle-focused)")
    print("  3. Generate images with the graphic designer agent")
    print("  4. Create integrated social media posts with matching images")
    print("  5. Verify and improve content using the verify_content tool")
    print("  6. Ask follow-up questions (the system maintains conversation memory)")
    print()
    print("Example queries:")
    print("  - What's the weather like in Tokyo today?")
    print("  - Create a LinkedIn post about AI advancements in healthcare")
    print("  - Create an Instagram caption for a new coffee shop opening")
    print("  - Generate an image of a futuristic cityscape with flying cars")
    print("  - Create a LinkedIn post with a matching image about renewable energy")
    print("  - Write a paragraph about climate change and verify it using the verify_content tool")
    print("  - Create content and use the verification tool to ensure quality")
    print()
    
    # Enable direct verification by default
    use_direct_verification = True
    print("\nDirect verification mode enabled. All responses will be verified automatically.")
    
    # Initialize conversation history
    conversation_history = []
    
    while True:
        user_input = input("\nEnter your query (or 'exit' to quit): ")
        
        if user_input.lower() == 'exit':
            break
            
        print("\nProcessing your request...\n")
        
        # Create the input by combining conversation history with new user input
        if conversation_history:
            # Add the new user message to the conversation history
            current_input = conversation_history + [{"role": "user", "content": user_input}]
        else:
            # First message in the conversation
            current_input = user_input
        
        # Determine if we should use direct verification
        if use_direct_verification:
            # Use direct verification with tracing
            with trace("Multi-agent with direct verification"):
                print("Running with direct verification...")
                final_output, verification_history = await verification.verify_and_improve(
                    orchestrator_agent,
                    current_input,
                    trace_name="Direct orchestrator verification"
                )
                
                # Create a result-like object for the conversation history
                if conversation_history:
                    conversation_history = conversation_history + [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": final_output}
                    ]
                else:
                    conversation_history = [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": final_output}
                    ]
                
                # Show verification details
                print("\nVerification process:")
                for i, attempt in enumerate(verification_history):
                    print(f"Attempt {i+1}:")
                    print(f"Score: {attempt.get('score', 'N/A')}")
                    print(f"Feedback: {attempt.get('feedback', 'N/A')}")
                    print("-" * 40)
                
                print(f"\nFinal verified response:\n{final_output}\n")
        else:
            # Run the orchestration with tracing (tool-based verification)
            with trace("Multi-agent orchestration"):
                result = await Runner.run(orchestrator_agent, current_input)
                
            # Update conversation history for the next turn
            conversation_history = result.to_input_list()
            
            print(f"\nFinal response:\n{result.final_output}\n")
        
        print("-" * 60)

if __name__ == "__main__":
    asyncio.run(main())