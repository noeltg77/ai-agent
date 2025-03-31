"""
Multi-Agent Manager that handles multiple agent interactions and session management
"""
from __future__ import annotations

import asyncio
import os
import time
import threading
from typing import List, Optional, Dict, Any, Tuple, Union

# Set OpenAI API key explicitly before importing more OpenAI-related modules
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

from agents import Agent, Runner, trace, ItemHelpers, gen_trace_id, WebSearchTool
from agents.mcp import MCPServerStdio

from src.tools import get_current_time, calculate_days_between, format_data, generate_image, get_todays_date
from src.prompt_loader import PromptLoader
from src.verification_sdk import Verification

# Import this dynamically to avoid circular imports
from API.session import AgentSession

class MultiAgentManager:
    def __init__(self, user_location: Dict[str, Any] = None, verification_enabled: bool = True, max_verification_attempts: int = 3):
        # Set default user location if none provided
        if user_location is None:
            user_location = {"type": "approximate", "city": "San Francisco"}
            
        # Initialize the prompt loader
        PromptLoader.initialize()
        
        # Initialize verification system
        self.verification_enabled = verification_enabled
        self.verification = Verification(max_attempts=max_verification_attempts)
        
        # Initialize the Replicate Designer MCP server
        self.replicate_designer_mcp = MCPServerStdio(
            name="Replicate Designer MCP",
            params={
                "command": "npx",
                "args": ["-y", "github:yourusername/replicate-designer"],
                "env": {
                    "REPLICATE_API_TOKEN": os.environ.get("REPLICATE_API_TOKEN", "")
                }
            },
            cache_tools_list=True
        )
        
        # Store for API sessions
        self.sessions: Dict[str, AgentSession] = {}
        self.session_lock = threading.Lock()
        self.session_cleanup_interval = 60 * 60  # 1 hour
        self.session_max_age = 24 * 60 * 60  # 24 hours
        self.session_inactivity_timeout = 2 * 60 * 60  # 2 hours
        
        # Start session cleanup thread
        self._start_session_cleanup()
            
        # Create specialized agents
        self.research_agent = Agent(
            name="research_agent",
            instructions=PromptLoader.get_prompt("research_agent"),
            tools=[WebSearchTool(user_location=user_location)]
        )
        
        # Create platform-specific social media agents
        self.linkedin_agent = Agent(
            name="linkedin_agent",
            instructions=PromptLoader.get_prompt("linkedin_agent"),
        )
        
        self.instagram_agent = Agent(
            name="instagram_agent",
            instructions=PromptLoader.get_prompt("instagram_agent"),
        )
        
        # Create the social media orchestrator agent that manages platform-specific agents
        self.social_media_agent = Agent(
            name="social_media_agent",
            instructions=PromptLoader.get_prompt("social_media_agent"),
            tools=[
                self.linkedin_agent.as_tool(
                    tool_name="create_linkedin_content",
                    tool_description="Create professional content optimized for LinkedIn's business-focused audience.",
                ),
                self.instagram_agent.as_tool(
                    tool_name="create_instagram_content",
                    tool_description="Create visually-oriented content optimized for Instagram's lifestyle-focused platform.",
                ),
            ],
        )
        
        self.summarizer_agent = Agent(
            name="summarizer_agent",
            instructions=PromptLoader.get_prompt("summarizer_agent"),
        )
        
        # Create the graphic designer agent with only the MCP tool
        self.graphic_designer_agent = Agent(
            name="graphic_designer_agent",
            instructions=PromptLoader.get_prompt("graphic_designer_agent"),
            tools=[],  # Remove the local generate_image tool
            mcp_servers=[self.replicate_designer_mcp],  # Only use the MCP server for image generation
        )
        
        # Create orchestrator agent with the specialized agents as tools
        self.orchestrator_agent = Agent(
            name="orchestrator_agent",
            instructions=PromptLoader.get_prompt("orchestrator_agent"),
            tools=[
                # Agent tools
                self.research_agent.as_tool(
                    tool_name="research",
                    tool_description="Use this tool to search for factual information or answer knowledge-based questions.",
                ),
                self.social_media_agent.as_tool(
                    tool_name="create_social_media_content",
                    tool_description="Use this tool to generate optimized content for different social media platforms based on the target audience and goals.",
                ),
                self.graphic_designer_agent.as_tool(
                    tool_name="create_image",
                    tool_description="Use this tool to generate custom images based on detailed descriptions. Useful for creating visuals for social media posts.",
                ),
                self.summarizer_agent.as_tool(
                    tool_name="summarize",
                    tool_description="Use this tool to create concise summaries of research and social media content.",
                ),
                
                # Verification tool from our SDK-style module
                self.verification.get_verify_function_tool(),
                
                # Function tools
                get_current_time,
                get_todays_date,
                calculate_days_between,
                format_data,
            ],
        )
    
    def _start_session_cleanup(self):
        """Start a background thread to clean up old sessions"""
        def cleanup_worker():
            while True:
                time.sleep(self.session_cleanup_interval)
                self._cleanup_sessions()
        
        # Start the cleanup thread as a daemon
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_sessions(self):
        """Remove expired sessions"""
        current_time = time.time()
        to_remove = []
        
        with self.session_lock:
            for session_id, session in self.sessions.items():
                # Remove sessions that are too old
                if current_time - session.created_at > self.session_max_age:
                    to_remove.append(session_id)
                    continue
                
                # Remove sessions that have been inactive for too long
                if current_time - session.last_active > self.session_inactivity_timeout:
                    to_remove.append(session_id)
            
            # Remove the expired sessions
            for session_id in to_remove:
                del self.sessions[session_id]
        
        if to_remove:
            print(f"Cleaned up {len(to_remove)} expired sessions")
    
    def get_or_create_session(self, session_id: str) -> AgentSession:
        """Get an existing session or create a new one"""
        with self.session_lock:
            if session_id in self.sessions:
                # Update the last active time
                self.sessions[session_id].touch()
                return self.sessions[session_id]
            
            # Create a new session
            session = AgentSession(session_id=session_id)
            self.sessions[session_id] = session
            return session

    async def __aenter__(self):
        """Async context manager entry - connects to MCP servers"""
        await self.replicate_designer_mcp.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_value, traceback):
        """Async context manager exit - disconnects from MCP servers"""
        await self.replicate_designer_mcp.__aexit__(exc_type, exc_value, traceback)
    
    async def process_query(self, query: str, conversation_history=None, use_verification: bool = None) -> tuple:
        """
        Process a user query through the multi-agent system.
        
        Args:
            query: The user's input query
            conversation_history: Optional list of previous conversation items
            use_verification: Override the default verification setting for this specific query
            
        Returns:
            A tuple containing (final_output, updated_conversation_history, verification_history)
        """
        # Create a unique trace ID for this process
        trace_id = gen_trace_id()
        verification_history = None
        
        # Determine if verification should be used for this query
        should_verify = self.verification_enabled if use_verification is None else use_verification
        
        # Create a trace for the entire process
        with trace("Multi-agent process", trace_id=trace_id):
            print(f"Trace ID: {trace_id}")
            print("Processing query with orchestrator agent...")
            
            # Prepare the input with conversation history if available
            if conversation_history:
                # Add the new user message to the conversation history
                current_input = conversation_history + [{"role": "user", "content": query}]
            else:
                # First message in the conversation
                current_input = query
            
            # Log verification status
            if should_verify:
                print("Verification enabled: Content will be verified and improved if needed")
                
                # Run the orchestrator with verification using our SDK-style implementation
                final_output, verification_history = await self.verification.verify_and_improve(
                    self.orchestrator_agent,
                    query if not conversation_history else current_input,
                    trace_name=f"Query verification process"
                )
                
                # Create a result-like object for the conversation history
                # We need to get the conversation history from the last verification attempt
                last_attempt_output = verification_history[-1]["output"]
                
                if conversation_history:
                    updated_history = conversation_history + [
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": last_attempt_output}
                    ]
                else:
                    updated_history = [
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": last_attempt_output}
                    ]
                
                return final_output, updated_history, verification_history
            else:
                # When not using verification, just run the agent with default trace
                result = await Runner.run(
                    self.orchestrator_agent,
                    current_input,
                )
                
                # Return the final output and the updated conversation history
                return result.final_output, result.to_input_list(), None

    async def create_verified_social_media_content(self, query: str, platform: str = None) -> Dict[str, Any]:
        """
        Create verified social media content for a specific platform or all platforms.
        
        Args:
            query: The user's request for social media content
            platform: Optional platform specification ("linkedin", "instagram", or None for both)
            
        Returns:
            Dictionary with the generated content and verification history
        """
        # Determine which agent to use based on the platform
        if platform and platform.lower() == "linkedin":
            input_query = f"Create a LinkedIn post about: {query}"
            agent = self.linkedin_agent
            platform_name = "LinkedIn"
        elif platform and platform.lower() == "instagram":
            input_query = f"Create an Instagram post about: {query}"
            agent = self.instagram_agent
            platform_name = "Instagram"
        else:
            input_query = f"Create social media content about: {query}"
            agent = self.social_media_agent
            platform_name = "multiple platforms"
        
        print(f"Generating verified social media content for {platform_name}...")
        
        # Run the content creation with verification
        # The verify_social_media_content method creates its own trace
        final_content, verification_history = await self.verification.verify_social_media_content(
            agent,
            input_query
        )
        
        # Prepare and return the results
        return {
            "content": final_content,
            "platform": platform_name,
            "verification_history": verification_history,
            "verification_attempts": len(verification_history),
            "original_query": query
        }
        
    async def create_verified_image(self, prompt: str, aspect_ratio: str = "1:1") -> Dict[str, Any]:
        """
        Create a verified image using the graphic designer agent.
        
        Args:
            prompt: The description of the image to generate
            aspect_ratio: The aspect ratio for the image
            
        Returns:
            Dictionary with the generated image URL and verification history
        """
        # Prepare the input query
        input_query = f"Generate an image of {prompt}. Use aspect ratio {aspect_ratio}."
        print(f"Generating verified image for: {prompt}...")
        
        # Run the image generation with verification
        # The verify_image_generation method creates its own trace
        final_result, verification_history = await self.verification.verify_image_generation(
            self.graphic_designer_agent,
            input_query
        )
        
        # Check if we have a valid image URL
        has_image_url = isinstance(final_result, str) and final_result.startswith("http")
        
        # Prepare and return the results
        return {
            "image_result": final_result,
            "verification_history": verification_history,
            "verification_attempts": len(verification_history),
            "original_prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "success": has_image_url
        }
    
    async def run_parallel_summary(self, content: str, num_summaries: int = 3) -> str:
        """
        Run multiple summarizers in parallel and pick the best one.
        
        Args:
            content: The content to summarize
            num_summaries: Number of parallel summaries to generate
            
        Returns:
            The selected best summary
        """
        print(f"Generating {num_summaries} parallel summaries...")
        
        # Generate tasks for parallel execution
        tasks = [
            asyncio.create_task(
                Runner.run(
                    self.summarizer_agent,
                    f"Summarize the following content:\n\n{content}"
                )
            )
            for _ in range(num_summaries)
        ]
        
        # Execute all tasks in parallel
        summaries = await asyncio.gather(*tasks)
        
        # Create a simple agent to select the best summary
        selector_agent = Agent(
            name="selector_agent",
            instructions="You evaluate multiple summaries and select the best one based on clarity, accuracy, and conciseness.",
        )
        
        # Format the summaries for the selector agent
        formatted_summaries = "\n\n".join([
            f"Summary {i+1}:\n{summary.final_output}"
            for i, summary in enumerate(summaries)
        ])
        
        input_for_selector = f"Original content:\n{content}\n\nPlease select the best summary from the following options:\n\n{formatted_summaries}"
        
        # Get the best summary
        selection_result = await Runner.run(
            selector_agent,
            input_for_selector,
        )
        
        return selection_result.final_output