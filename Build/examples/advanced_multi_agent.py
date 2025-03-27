import asyncio
import os
import sys

# Add the project root directory to the Python path for importing local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.multi_agent_manager import MultiAgentManager
from src.prompt_loader import PromptLoader

# Initialize the prompt loader
PromptLoader.initialize()

async def run_orchestrator_demo():
    """
    Demo the basic orchestrator pattern.
    """
    manager = MultiAgentManager()
    
    print("\n===== ORCHESTRATOR PATTERN DEMO =====\n")
    print("This demo shows how the orchestrator agent coordinates specialized agents.")
    print("Try asking questions that might need research, creativity, or both.\n")
    print("This demo includes conversation memory - the agent will remember previous exchanges.\n")
    
    # Initialize conversation history
    conversation_history = None
    
    while True:
        query = input("Enter your query (or 'exit' to move to next demo): ")
        
        if query.lower() == 'exit':
            break
            
        print("\nProcessing request...\n")
        
        # Process the query and get updated conversation history
        response, conversation_history = await manager.process_query(query, conversation_history)
        
        print("\n----- RESPONSE -----\n")
        print(response)
        print("\n---------------------\n")

async def run_parallel_demo():
    """
    Demo the parallelization pattern with multiple summarizers.
    """
    manager = MultiAgentManager()
    
    print("\n===== PARALLEL PROCESSING DEMO =====\n")
    print("This demo shows how we can run multiple agents in parallel and select the best result.")
    print("You'll provide content to summarize, and we'll generate multiple summaries in parallel.\n")
    
    content = input("Enter the content you'd like to summarize (or 'exit' to quit): ")
    
    if content.lower() == 'exit':
        return
        
    print("\nGenerating multiple summaries in parallel...\n")
    best_summary = await manager.run_parallel_summary(content)
    
    print("\n----- BEST SUMMARY -----\n")
    print(best_summary)
    print("\n------------------------\n")

async def run_tools_demo():
    """
    Demo the usage of function tools.
    """
    manager = MultiAgentManager()
    
    print("\n===== FUNCTION TOOLS DEMO =====\n")
    print("This demo shows how the orchestrator can use function tools for utility operations.")
    print("Try asking questions that might use date/time functions or data formatting.\n")
    print("Example queries:")
    print("- What is the current time?")
    print("- How many days between 2023-01-01 and 2023-12-31?")
    print("- Format this data as json: {\"name\": \"John\", \"age\": 30, \"city\": \"New York\"}")
    print("- What's happening in world news today? (uses web search)")
    print("- Try follow-up questions to test conversation memory\n")
    
    # Initialize conversation history
    conversation_history = None
    
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        
        if query.lower() == 'exit':
            break
            
        print("\nProcessing request...\n")
        
        # Process the query and get updated conversation history
        response, conversation_history = await manager.process_query(query, conversation_history)
        
        print("\n----- RESPONSE -----\n")
        print(response)
        print("\n---------------------\n")

async def main():
    print("Welcome to the Multi-Agent System Demo")
    print("======================================")
    print("This demonstration showcases three multi-agent patterns:")
    print("1. The orchestrator pattern - where a central agent coordinates specialized agents")
    print("2. The parallelization pattern - where we run multiple agents in parallel and select the best result")
    print("3. The function tools pattern - where we use Python functions as tools for the agent")
    print("\nLet's begin with the orchestrator pattern...\n")
    
    await run_orchestrator_demo()
    await run_parallel_demo()
    await run_tools_demo()
    
    print("\nThank you for trying the Multi-Agent System Demo!")

if __name__ == "__main__":
    asyncio.run(main())