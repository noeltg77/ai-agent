"""
Session management for the multi-agent API.
"""
from __future__ import annotations

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from agents import Agent, WebSearchTool, function_tool

from src.prompt_loader import PromptLoader
from src.tools import generate_image
from src.verification_sdk import Verification

@dataclass
class AgentSession:
    """
    A session for the multi-agent system, managing conversation state and agent instances.
    """
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Lazy-loaded agent instances
    _research_agent: Optional[Agent] = None
    _linkedin_agent: Optional[Agent] = None
    _instagram_agent: Optional[Agent] = None
    _social_media_agent: Optional[Agent] = None
    _graphic_designer_agent: Optional[Agent] = None
    _summarizer_agent: Optional[Agent] = None
    _orchestrator_agent: Optional[Agent] = None
    _verification: Optional[Verification] = None
    
    def touch(self):
        """Update the last active timestamp."""
        self.last_active = time.time()
    
    @property
    def age(self) -> float:
        """Get the age of the session in seconds."""
        return time.time() - self.created_at
    
    @property
    def inactive_time(self) -> float:
        """Get the time since last activity in seconds."""
        return time.time() - self.last_active
    
    @property
    def research_agent(self) -> Agent:
        """Lazy-load the research agent."""
        if self._research_agent is None:
            self._research_agent = Agent(
                name=f"research_agent_{self.session_id}",
                instructions=PromptLoader.get_prompt("research_agent"),
                tools=[WebSearchTool(user_location={"type": "approximate", "city": "London"})]
            )
        return self._research_agent
    
    @property
    def linkedin_agent(self) -> Agent:
        """Lazy-load the LinkedIn agent."""
        if self._linkedin_agent is None:
            self._linkedin_agent = Agent(
                name=f"linkedin_agent_{self.session_id}",
                instructions=PromptLoader.get_prompt("linkedin_agent"),
            )
        return self._linkedin_agent
    
    @property
    def instagram_agent(self) -> Agent:
        """Lazy-load the Instagram agent."""
        if self._instagram_agent is None:
            self._instagram_agent = Agent(
                name=f"instagram_agent_{self.session_id}",
                instructions=PromptLoader.get_prompt("instagram_agent"),
            )
        return self._instagram_agent
    
    @property
    def social_media_agent(self) -> Agent:
        """Lazy-load the social media agent."""
        if self._social_media_agent is None:
            self._social_media_agent = Agent(
                name=f"social_media_agent_{self.session_id}",
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
        return self._social_media_agent
    
    @property
    def graphic_designer_agent(self) -> Agent:
        """Lazy-load the graphic designer agent."""
        if self._graphic_designer_agent is None:
            self._graphic_designer_agent = Agent(
                name=f"graphic_designer_agent_{self.session_id}",
                instructions=PromptLoader.get_prompt("graphic_designer_agent"),
                tools=[generate_image],
            )
        return self._graphic_designer_agent
    
    @property
    def summarizer_agent(self) -> Agent:
        """Lazy-load the summarizer agent."""
        if self._summarizer_agent is None:
            self._summarizer_agent = Agent(
                name=f"summarizer_agent_{self.session_id}",
                instructions=PromptLoader.get_prompt("summarizer_agent"),
            )
        return self._summarizer_agent
    
    @property
    def verification(self) -> Verification:
        """Lazy-load the verification system."""
        if self._verification is None:
            self._verification = Verification(max_attempts=2)
        return self._verification
    
    @property
    def orchestrator_agent(self) -> Agent:
        """Lazy-load the orchestrator agent with all tools."""
        if self._orchestrator_agent is None:
            self._orchestrator_agent = Agent(
                name=f"orchestrator_agent_{self.session_id}",
                instructions=PromptLoader.get_prompt("orchestrator_agent"),
                tools=[
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
                    # Add the verification tool
                    self.verification.get_verify_function_tool(),
                ],
            )
        return self._orchestrator_agent
