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

# Check for Replicate API token
replicate_api_token = os.getenv("REPLICATE_API_TOKEN")
if replicate_api_token:
    print(f"Replicate API Token set: {replicate_api_token[:5]}...{replicate_api_token[-4:]}")
else:
    print("Warning: No Replicate API token found in .env file. MCP image generation will not work.")

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents import Agent, Runner, trace, gen_trace_id
from agents.mcp import MCPServerStdio
from src.prompt_loader import PromptLoader
from src.tools import generate_image

# Initialize the prompt loader
PromptLoader.initialize()

async def main():
    print("MCP Image Generation Example")
    print("============================")
    print("This example demonstrates using the Replicate Designer MCP server")
    print("for high-quality image generation with the graphic designer agent.")
    print()
    
    # Initialize the Replicate Designer MCP server
    async with MCPServerStdio(
        name="Replicate Designer MCP",
        params={
            "command": "npx",
            "args": ["-y", "github:yourusername/replicate-designer"],
            "env": {
                "REPLICATE_API_TOKEN": os.environ.get("REPLICATE_API_TOKEN", "")
            }
        },
        cache_tools_list=True
    ) as replicate_designer_mcp:
        # Create the graphic designer agent with only the MCP tool
        graphic_designer_agent = Agent(
            name="graphic_designer_agent",
            instructions=PromptLoader.get_prompt("graphic_designer_agent"),
            tools=[],  # Remove the local generate_image tool
            mcp_servers=[replicate_designer_mcp],  # Only use the MCP server for image generation
        )
        
        # Create a trace for the entire process
        trace_id = gen_trace_id()
        with trace("MCP Image Generation Demo", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/{trace_id}")
            
            print("\nThe graphic designer agent now has access to:")
            print("MCP generate_image tool (using Replicate's Flux 1.1 Pro model via MCP)")
            print("\nExample prompts:")
            print("- Generate an image of a futuristic cityscape with flying cars")
            print("- Create a photorealistic portrait of a robot reading a book")
            print("- Design a logo for an AI startup with blue and purple colors")
            print("- Create an image of a mountain landscape at sunset with high quality using the MCP tool")
            print()
            
            while True:
                user_input = input("\nEnter your image prompt (or 'exit' to quit): ")
                
                if user_input.lower() == 'exit':
                    break
                    
                # Run the agent with the user's prompt
                print("\nProcessing your request...\n")
                result = await Runner.run(
                    graphic_designer_agent,
                    f"Generate an image based on this description: {user_input}"
                )
                
                # Show the result
                print("\nGraphic Designer Agent Response:")
                print(result.final_output)
                print("\n" + "-" * 60)

if __name__ == "__main__":
    asyncio.run(main())