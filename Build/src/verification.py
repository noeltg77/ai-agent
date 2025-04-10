"""
Verification module for implementing the "LLM as a Judge" pattern.
This allows us to verify and improve outputs generated by other agents.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, Optional, Any, Dict, List, Tuple, Union
from pydantic import BaseModel

from agents import Agent, Runner, TResponseInputItem, ItemHelpers, trace, gen_trace_id
from agents.tracing import agent_span, custom_span, Trace

from src.prompt_loader import PromptLoader

class EvaluationFeedback(BaseModel):
    """
    Output model for the verifier agent.
    """
    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]
    

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
        self.verifier_prompt = PromptLoader.get_prompt("verifier_agent")
        self.max_attempts = max_attempts

    # Define the verification function at the class level
    from agents import function_tool

    @function_tool(
        name_override="verify_content",
        description_override="Verify and improve content using the LLM as a Judge pattern. IMPORTANT: Use this tool for all content that needs to be checked for accuracy or quality."
    )
    async def verify_content_tool(self, content: str, content_type: str = "general") -> str:
        """
        Verify and improve content using the LLM as a Judge pattern.
        
        Args:
            content: The content to verify and improve
            content_type: The type of content (general, social_media, or image)
            
        Returns:
            The verified and improved content
        """
        # Create a verification agent for this specific verification
        verification_id = gen_trace_id()[:8]
        verifier_agent = Agent(
            name=f"verifier_agent_{verification_id}",
            instructions=self.verifier_prompt,
            output_type=EvaluationFeedback,
        )
        
        # Log that the verification tool is being used
        print(f"🔍 Verification tool called for content type: {content_type}")
        print(f"🔍 Content length: {len(content)} characters")
        
        # Check if we're in a function tool span by accessing the current span
        from agents.tracing import get_current_span
        current_span = get_current_span()
        if current_span:
            print(f"🔍 Current span: {current_span.name}")
            print(f"🔍 Current span type: {type(current_span)}")
            print(f"🔍 In function span: {current_span.name.startswith('function')}")
        else:
            print("🔍 No current span found")
        
        # Create a trace specifically for this verification tool use
        trace_id = gen_trace_id()
        print(f"🔍 Verification tool trace ID: {trace_id}")
        
        with trace(f"Content verification tool - {content_type}", trace_id=trace_id):
            # Run the verification process
            with agent_span(verifier_agent):
                verification_input = f"Content to verify: {content}\n\nPlease evaluate this content."
                
                # This will be captured properly in the trace as an agent run
                verifier_result = await Runner.run(
                    verifier_agent, 
                    verification_input
                )
            
            # Get the evaluation feedback
            evaluation: EvaluationFeedback = verifier_result.final_output
            
            # If the content passes verification, return it as is
            if evaluation.score == "pass":
                return f"✅ VERIFIED: {content}"
            
            # If the content needs improvement, add a improvement request
            input_items = [
                {"content": content, "role": "user"},
                {"content": f"Please improve this based on the following feedback: {evaluation.feedback}", "role": "user"}
            ]
            
            # Create a content improver agent
            improver_agent = Agent(
                name=f"content_improver_{verification_id}",
                instructions=f"You improve content based on feedback. Content type: {content_type}."
            )
            
            # Run the improver agent
            with agent_span(improver_agent):
                improved_result = await Runner.run(
                    improver_agent,
                    input_items
                )
            
            # Return the improved content
            return f"🔄 IMPROVED: {improved_result.final_output}\n\nOriginal feedback: {evaluation.feedback}"
    
    def get_verify_function_tool(self):
        """
        Gets the verification function tool.
        
        Returns:
            The function tool for verification
        """
        # Return a reference to the class method
        return self.verify_content_tool
        
    async def verify_and_improve(
        self, 
        agent: Agent, 
        input_query: str,
        context: Optional[Any] = None,
        trace_name: str = "Content verification"
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Run an agent, verify its output, and improve it if necessary.
        
        Args:
            agent: The agent to run and verify
            input_query: The user's input query
            context: Optional context for the agent run
            trace_name: Name for the trace
            
        Returns:
            Tuple of (final output, improvement history)
        """
        input_items: List[TResponseInputItem] = [{"content": input_query, "role": "user"}]
        improvement_history = []
        
        # Create unique IDs for this verification process
        verification_id = gen_trace_id()
        main_trace = None
        
        # Get agent name for better logs
        agent_name = getattr(agent, 'name', 'unnamed_agent')
        
        # Create a dedicated verifier agent with a unique instance for tracing
        verifier_instance_id = gen_trace_id()[:8]  # Shorter ID for readability
        verifier_agent = Agent(
            name=f"verifier_agent_{verifier_instance_id}",
            instructions=self.verifier_prompt,
            output_type=EvaluationFeedback,
        )
        
        latest_output = None
        attempts = 0
        original_output = None
        
        # Create and start the main trace for the entire verification process
        main_trace = Trace(
            workflow_name=f"LLM as a Judge: {trace_name}",
            trace_id=verification_id
        )
        
        try:
            # Start the main trace
            main_trace.start(mark_as_current=True)
            
            # Log verification setup as a custom span
            with custom_span("verification_setup", {"agent": agent_name, "max_attempts": self.max_attempts}):
                # The span will capture the setup context
                pass
            
            # Run the verification loop
            while attempts < self.max_attempts:
                current_attempt = attempts + 1
                
                # Add a custom span for this attempt
                with custom_span(f"verification_attempt", 
                                {"attempt": current_attempt, "max_attempts": self.max_attempts}):
                    
                    # Run the original agent with explicit agent span
                    with agent_span(agent):
                        # This will be captured properly in the trace as an agent run
                        agent_result = await Runner.run(
                            agent,
                            input_items,
                            context=context
                        )
                    
                    # Get the latest output
                    input_items = agent_result.to_input_list()
                    latest_output = agent_result.final_output
                    
                    # For the first run, capture the original output
                    if attempts == 0:
                        original_output = latest_output
                    
                    # Record this attempt in the history
                    improvement_history.append({
                        "attempt": current_attempt,
                        "output": latest_output,
                    })
                    
                    # Run the verifier agent with explicit agent span
                    with agent_span(verifier_agent):
                        # Prepare verification input
                        verification_input = f"Original request: {input_query}\n\nGenerated content: {latest_output}\n\nPlease evaluate this content."
                        
                        # This will be captured properly in the trace as an agent run
                        verifier_result = await Runner.run(
                            verifier_agent, 
                            verification_input
                        )
                    
                    # Get the evaluation feedback
                    evaluation: EvaluationFeedback = verifier_result.final_output
                    
                    # Record the feedback in the history
                    improvement_history[-1]["feedback"] = evaluation.feedback
                    improvement_history[-1]["score"] = evaluation.score
                    
                    # Create a custom span for the decision process
                    with custom_span("verification_decision", {
                        "attempt": current_attempt,
                        "score": evaluation.score,
                        "feedback_summary": evaluation.feedback[:50] + "..." if len(evaluation.feedback) > 50 else evaluation.feedback
                    }):
                        # If the content passes verification, return it
                        if evaluation.score == "pass":
                            with custom_span("verification_completed", {
                                "result": "pass",
                                "attempts_needed": current_attempt
                            }):
                                pass
                            return latest_output, improvement_history
                        
                        # If we've reached the maximum attempts, break
                        if current_attempt >= self.max_attempts:
                            with custom_span("verification_completed", {
                                "result": "max_attempts_reached",
                                "final_score": evaluation.score
                            }):
                                pass
                            break
                        
                        # Add the feedback to the input for the next attempt
                        input_items.append({
                            "content": f"Please improve this based on the following feedback: {evaluation.feedback}", 
                            "role": "user"
                        })
                
                attempts += 1
            
            # Add a final span for the verification process summary
            with custom_span("verification_summary", {
                "attempts": attempts,
                "improvement_history_length": len(improvement_history),
                "final_score": improvement_history[-1]["score"] if improvement_history else "unknown"
            }):
                pass
            
            # Return the latest output and improvement history
            return latest_output, improvement_history
        
        finally:
            # Make sure to properly finish the trace
            if main_trace:
                main_trace.finish(reset_current=True)
        
    async def verify_social_media_content(
        self,
        social_media_agent: Agent,
        input_query: str,
        context: Optional[Any] = None
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Specifically verify and improve social media content.
        
        Args:
            social_media_agent: The social media agent to run
            input_query: The user's input query
            context: Optional context for the agent run
            
        Returns:
            Tuple of (final social media content, improvement history)
        """
        # Extract platform info for better trace naming
        platform_type = "generic"
        if "linkedin" in input_query.lower():
            platform_type = "LinkedIn"
        elif "instagram" in input_query.lower():
            platform_type = "Instagram"
        
        trace_name = f"Social media content - {platform_type}"
        
        # The verify_and_improve method creates its own trace, so we don't need to create one here
        # Just add extra context via the trace_name parameter
        
        # Use the standard verification process with the specialized trace name
        result = await self.verify_and_improve(
            social_media_agent,
            input_query,
            context,
            trace_name=trace_name
        )
                
        return result
        
    async def verify_image_generation(
        self,
        graphic_designer_agent: Agent,
        input_query: str,
        context: Optional[Any] = None
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Specifically verify and improve image generation prompts.
        
        Args:
            graphic_designer_agent: The graphic designer agent to run
            input_query: The user's input query
            context: Optional context for the agent run
            
        Returns:
            Tuple of (final image URL or response, improvement history)
        """
        # Extract aspect ratio info for better trace naming
        aspect_ratio = "1:1"  # Default
        aspect_matches = ["1:1", "16:9", "9:16", "4:3", "3:4"]
        for aspect in aspect_matches:
            if aspect in input_query:
                aspect_ratio = aspect
                break
                
        trace_name = f"Image generation - {aspect_ratio}"
        
        # The verify_and_improve method creates its own trace, so we don't need to create one here
        # Just add extra context via the trace_name parameter
        
        # Use the standard verification process with the specialized trace name
        result = await self.verify_and_improve(
            graphic_designer_agent,
            input_query,
            context,
            trace_name=trace_name
        )
        
        return result