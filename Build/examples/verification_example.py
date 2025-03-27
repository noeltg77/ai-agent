"""
Verification Example - Demonstrates the LLM as a Judge pattern
"""
import asyncio
import os
import sys
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents import gen_trace_id, trace
from agents.tracing import Trace, custom_span
from src.multi_agent_manager import MultiAgentManager
from src.prompt_loader import PromptLoader

# Initialize the prompt loader
PromptLoader.initialize()

async def run_verification_demo():
    """
    Demo the verification capabilities.
    """
    # Generate a trace ID for the entire demo
    main_trace_id = gen_trace_id()
    
    print("\n===== VERIFICATION DEMO =====\n")
    print(f"Main Trace ID: {main_trace_id}")
    print("This demo shows how content verification improves agent outputs through multiple iterations.")
    print("The system will generate content, verify it, and improve it until it meets quality standards.")
    print("All verification activities will be visible in the trace logs under this trace ID.\n")
    
    # Create the main trace for the entire demo using the trace() function
    demo_trace = trace(
        workflow_name="LLM as a Judge Demo",
        trace_id=main_trace_id
    )
    
    # Start the main trace - this is automatically done when using it as a context manager
    # but we'll use it explicitly here
    demo_trace.start(mark_as_current=True)
    
    try:
        # Create a multi-agent manager with verification enabled
        with custom_span("demo_initialization", {"max_verification_attempts": 3}):
            manager = MultiAgentManager(verification_enabled=True, max_verification_attempts=3)
        
        # Demo for general queries
        with custom_span("general_query_section", {}):
            print("\n----- GENERAL QUERY VERIFICATION -----\n")
            query = input("Enter a general query to process with verification (or press Enter for default): ")
            
            if not query:
                query = "Explain how quantum computing will impact cybersecurity in the next decade"
                print(f"Using default query: {query}")
            
            print("\nProcessing with verification enabled (may take a moment)...\n")
            
            # Process the query with verification - creates its own trace
            response, history, verification_info = await manager.process_query(
                query, 
                use_verification=True
            )
            
            # Print the verification process
            print("\nVerification Process:\n")
            for i, attempt in enumerate(verification_info):
                print(f"Attempt {i+1}:")
                print(f"Score: {attempt.get('score', 'N/A')}")
                print(f"Feedback: {attempt.get('feedback', 'N/A')}")
                print("-" * 40)
            
            print("\nFinal Response after Verification:\n")
            print(response)
        
        # Demo for social media content
        with custom_span("social_media_content_section", {}):
            print("\n\n----- SOCIAL MEDIA CONTENT VERIFICATION -----\n")
            social_query = input("Enter a topic for social media content (or press Enter for default): ")
            
            if not social_query:
                social_query = "The launch of a new eco-friendly smartphone"
                print(f"Using default topic: {social_query}")
            
            platform = input("Which platform? (linkedin/instagram/both, default=linkedin): ")
            
            if not platform:
                platform = "linkedin"
                
            print(f"\nGenerating verified social media content for {platform}...\n")
            
            # Generate verified social media content - creates its own trace
            result = await manager.create_verified_social_media_content(social_query, platform)
            
            # Print the verification process
            print("\nVerification Process:\n")
            for i, attempt in enumerate(result["verification_history"]):
                print(f"Attempt {i+1}:")
                print(f"Score: {attempt.get('score', 'N/A')}")
                print(f"Feedback: {attempt.get('feedback', 'N/A')}")
                print("-" * 40)
            
            print(f"\nFinal {result['platform']} Content After Verification:\n")
            print(result["content"])
        
        # Demo for image generation
        with custom_span("image_generation_section", {}):
            print("\n\n----- IMAGE GENERATION VERIFICATION -----\n")
            image_prompt = input("Enter an image description (or press Enter for default): ")
            
            if not image_prompt:
                image_prompt = "A futuristic smart city with flying cars and renewable energy sources"
                print(f"Using default image prompt: {image_prompt}")
            
            aspect = input("Enter aspect ratio (1:1, 16:9, 9:16, default=1:1): ")
            if not aspect:
                aspect = "1:1"
            
            print(f"\nGenerating verified image with {aspect} aspect ratio...\n")
            
            # Generate verified image - creates its own trace
            # Note: This will require a valid REPLICATE_API_TOKEN in the .env file
            image_result = await manager.create_verified_image(image_prompt, aspect)
            
            # Print the verification process
            print("\nImage Prompt Verification Process:\n")
            for i, attempt in enumerate(image_result["verification_history"]):
                print(f"Attempt {i+1}:")
                print(f"Score: {attempt.get('score', 'N/A')}")
                print(f"Feedback: {attempt.get('feedback', 'N/A')}")
                print("-" * 40)
            
            print("\nFinal Image Generation Result:\n")
            print(image_result["image_result"])
        
        # Print summary
        with custom_span("verification_demo_summary", {
            "general_query_attempts": len(verification_info) if verification_info else 0,
            "social_media_attempts": result['verification_attempts'] if 'verification_attempts' in result else 0,
            "image_generation_attempts": image_result['verification_attempts'] if 'verification_attempts' in image_result else 0
        }):
            print("\n----- VERIFICATION SUMMARY -----\n")
            print(f"General Query: {len(verification_info)} verification attempts")
            print(f"Social Media Content: {result['verification_attempts']} verification attempts")
            print(f"Image Generation: {image_result['verification_attempts']} verification attempts")
    
    finally:
        # Make sure to properly finish the main trace
        demo_trace.finish(reset_current=True)

async def test_verification_tool():
    """
    Direct test of the verification tool to ensure it's working properly.
    """
    print("\n===== VERIFICATION TOOL DIRECT TEST =====\n")
    
    # Initialize the prompt loader
    PromptLoader.initialize()
    
    # Create a verification instance
    verification = Verification(max_attempts=2)
    
    # Get the verification tool
    verify_tool = verification.get_verify_function_tool()
    
    # Print information about the tool
    print(f"Verification tool name: {getattr(verify_tool, 'name', 'unknown')}")
    print(f"Verification tool type: {type(verify_tool)}")
    
    # Try to directly use the tool
    print("\nDirectly testing verification tool...")
    result = await verify_tool("This is a test content that needs verification.", "general")
    
    print("\nVerification Result:")
    print(result)
    
    return result

async def run_tool_based_demo():
    """
    Demonstrate verification using the verify_content tool directly.
    """
    # Initialize the prompt loader
    PromptLoader.initialize()
    
    print("\n===== TOOL-BASED VERIFICATION DEMO =====\n")
    print("This demo shows how verification can be used as a tool by other agents.")
    
    # First, run a direct test of the verification tool
    test_result = await test_verification_tool()
    
    # Create the main trace for the demo using the trace() function
    demo_trace = trace(
        workflow_name="Verification Tool Demo",
        trace_id=gen_trace_id()
    )
    
    # Start the main trace
    demo_trace.start(mark_as_current=True)
    
    try:
        # Create a manager with verification tool
        manager = MultiAgentManager(verification_enabled=True, max_verification_attempts=3)
        
        # Create a simple test agent that will use the verification tool
        test_agent = Agent(
            name="test_agent",
            instructions="You are a test agent that creates content and then verifies it using the verify_content tool.",
            tools=[
                # Properly decorated tool
                manager.verification.get_verify_function_tool()
            ]
        )
        
        # Run the test agent
        print("\nRunning test agent with explicit verification tool usage...")
        with custom_span("test_agent_run", {}):
            result = await Runner.run(
                test_agent,
                "Generate a short paragraph about quantum computing and then verify it."
            )
        
        print("\nTest Agent Result:")
        print(result.final_output)
        
        # Now run a query through the orchestrator agent, which should use verification
        print("\nRunning query through orchestrator agent with verification tool...")
        with custom_span("orchestrator_agent_run", {}):
            # Provide explicit instructions to verify the content to encourage tool use
            orchestrator_result = await Runner.run(
                manager.orchestrator_agent,
                "Explain how artificial intelligence is changing healthcare. First, generate the explanation, and then use the verify_content tool to ensure the information is accurate and well-structured."
            )
        
        print("\nOrchestrator Agent Result:")
        print(orchestrator_result.final_output)
    
    finally:
        # Make sure to properly finish the main trace
        demo_trace.finish(reset_current=True)

async def main():
    print("Verification System Demo")
    print("=======================")
    print("This demonstration showcases the 'LLM as a Judge' pattern")
    print("where one agent creates content and another verifies and improves it.")
    print("The process continues until the content meets quality standards.")
    
    print("\nPart 1: Direct Verification Method")
    await run_verification_demo()
    
    print("\nPart 2: Tool-Based Verification Method")
    await run_tool_based_demo()
    
    print("\nThank you for trying the Verification System Demo!")

if __name__ == "__main__":
    asyncio.run(main())