"""
Example script demonstrating the use of the long form content creation system.

This example shows how to generate longform content using the specialized agents:
- Long Form Content Agent (orchestrator)
- Content Outliner Agent
- Copy Writer Agent
- Editor Agent

Usage:
    python longform_content_example.py
"""
import os
import asyncio
import argparse
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

# Set OpenAI API key explicitly for proper tracing and logging
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

# Import after setting API key
from agents import Agent, Runner, trace, gen_trace_id
from src.prompt_loader import PromptLoader
from src.multi_agent_manager import MultiAgentManager

async def generate_longform_content(topic: str, word_count: int = 1500, style: str = "informative"):
    """
    Generate longform content on the specified topic.
    
    Args:
        topic: The topic to write about
        word_count: Approximate number of words for the final content
        style: The writing style (e.g., informative, persuasive, conversational)
        
    Returns:
        The generated content and a summary of the process
    """
    print(f"Generating {word_count} word {style} content about: {topic}")
    
    # Initialize the PromptLoader
    PromptLoader.initialize()
    
    # Initialize the MultiAgentManager
    manager = MultiAgentManager()
    
    # Create a unique trace ID for this process
    trace_id = gen_trace_id()
    print(f"Trace ID: {trace_id}")
    
    # Create the trace for the entire process
    with trace("Longform content generation", trace_id=trace_id):
        # Connect to any MCP servers if available
        await manager.__aenter__()
        
        try:
            # Format the user request
            user_request = f"""
            Please create longform content about "{topic}".
            
            Requirements:
            - Target length: approximately {word_count} words
            - Style: {style}
            - Include a compelling introduction and conclusion
            - Use subheadings to organize the content
            - Back up claims with relevant examples or data
            """
            
            # Run the longform content agent
            print("Running long form content agent...")
            result = await Runner.run(
                manager.long_form_content_agent,
                user_request
            )
            
            # Get the final output
            final_content = result.final_output
            
            # Print URL to view the trace in the OpenAI platform
            print(f"View trace: https://platform.openai.com/traces/{trace_id}")
            
            return {
                "content": final_content,
                "topic": topic,
                "word_count": word_count,
                "style": style,
                "trace_id": trace_id
            }
            
        finally:
            # Ensure MCP servers are properly disconnected
            await manager.__aexit__(None, None, None)

def format_output(content_result):
    """Format the results for display."""
    print("\n" + "=" * 80)
    print(f"LONGFORM CONTENT: {content_result['topic']}")
    print(f"Style: {content_result['style']} | Target Word Count: {content_result['word_count']}")
    print("=" * 80)
    print(content_result['content'])
    print("\n" + "=" * 80)
    print(f"Trace ID: {content_result['trace_id']}")
    print(f"View trace: https://platform.openai.com/traces/{content_result['trace_id']}")

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate longform content using multi-agent system")
    parser.add_argument("--topic", type=str, default="The Future of Artificial Intelligence in Healthcare",
                        help="Topic for the longform content")
    parser.add_argument("--words", type=int, default=1500,
                        help="Target word count for the content")
    parser.add_argument("--style", type=str, default="informative",
                        choices=["informative", "persuasive", "conversational", "technical", "storytelling"],
                        help="Writing style for the content")
    args = parser.parse_args()
    
    # Generate the content
    content_result = await generate_longform_content(
        topic=args.topic,
        word_count=args.words,
        style=args.style
    )
    
    # Format and display the results
    format_output(content_result)

if __name__ == "__main__":
    asyncio.run(main())