"""
Verification module for implementing the "LLM as a Judge" pattern.
This module uses the OpenAI Agents SDK pattern for proper tracing.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Literal, Optional, Any, Dict, List, Tuple, Union

from agents import Agent, Runner, ItemHelpers, TResponseInputItem, trace, gen_trace_id, function_tool
from pydantic import BaseModel

from src.prompt_loader import PromptLoader

# Make sure we explicitly set the API key before importing more OpenAI stuff
api_key = os.getenv("OPENAI_API_KEY", "")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

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
        
    async def verify_and_improve(
        self, 
        agent: Agent, 
        input_query: str,
        context: Optional[Any] = None,
        trace_name: str = "Content verification"
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Run an agent, verify its output, and improve it if necessary.
        Uses the LLM as a Judge pattern directly from the SDK examples.
        
        Args:
            agent: The agent to run and verify
            input_query: The user's input query
            context: Optional context for the agent run
            trace_name: Name for the trace
            
        Returns:
            Tuple of (final output, improvement history)
        """
        # Create input items for the agent
        if isinstance(input_query, str):
            input_items: List[TResponseInputItem] = [{"content": input_query, "role": "user"}]
        else:
            # Assume it's already a list of input items
            input_items = input_query
            
        improvement_history = []
        
        # Create a verifier agent
        verification_id = gen_trace_id()[:8]  # Use a shorter ID for readability
        verifier_agent = Agent(
            name=f"verifier_agent_{verification_id}",
            instructions=self.verifier_prompt,
            output_type=EvaluationFeedback,
        )
        
        # Prepare for the verification loop
        latest_output = None
        attempts = 0
        
        # Run the verification loop within a trace
        with trace(trace_name):
            while attempts < self.max_attempts:
                # Run the content generator agent
                content_result = await Runner.run(
                    agent,
                    input_items,
                    context=context
                )
                
                # Get the latest output
                input_items = content_result.to_input_list()
                latest_output = content_result.final_output if hasattr(content_result, 'final_output') else ItemHelpers.text_message_outputs(content_result.new_items)
                
                # For the first run, capture the original output
                if attempts == 0:
                    original_output = latest_output
                
                # Record this attempt in the history
                improvement_history.append({
                    "attempt": attempts + 1,
                    "output": latest_output,
                })
                
                # Run the evaluator agent
                evaluator_result = await Runner.run(
                    verifier_agent, 
                    f"Original request: {input_query if isinstance(input_query, str) else 'User query'}\n\nGenerated content: {latest_output}\n\nPlease evaluate this content."
                )
                
                # Get the evaluation feedback
                evaluation: EvaluationFeedback = evaluator_result.final_output
                
                # Record the feedback in the history
                improvement_history[-1]["feedback"] = evaluation.feedback
                improvement_history[-1]["score"] = evaluation.score
                
                # If the content passes verification, return it
                if evaluation.score == "pass":
                    return latest_output, improvement_history
                
                # If we've reached the maximum attempts, break
                if attempts + 1 >= self.max_attempts:
                    break
                
                # Add the feedback to the input for the next attempt
                input_items.append({
                    "content": f"Please improve this based on the following feedback: {evaluation.feedback}", 
                    "role": "user"
                })
                
                attempts += 1
        
        # Return the latest output and improvement history
        return latest_output, improvement_history
    
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
        
        # Use the standard verification process with the specialized trace name
        return await self.verify_and_improve(
            social_media_agent,
            input_query,
            context,
            trace_name=f"Social media content - {platform_type}"
        )
    
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
                
        # Use the standard verification process with the specialized trace name
        return await self.verify_and_improve(
            graphic_designer_agent,
            input_query,
            context,
            trace_name=f"Image generation - {aspect_ratio}"
        )
    
    # Function tool for content verification
    async def _verify_content_impl(self, content: str, content_type: str) -> str:
        """Implementation of the verification function tool"""
        # Create a verification agent for this specific verification
        verification_id = gen_trace_id()[:8]
        verifier_agent = Agent(
            name=f"verifier_tool_{verification_id}",
            instructions=self.verifier_prompt,
            output_type=EvaluationFeedback,
        )
        
        # Run the verification within a trace
        with trace(f"Tool verification - {content_type}"):
            # Run the verifier to evaluate the content
            verifier_result = await Runner.run(
                verifier_agent,
                f"Content to verify: {content}\n\nPlease evaluate this content."
            )
            
            # Get the evaluation
            evaluation: EvaluationFeedback = verifier_result.final_output
            
            # If the content passes verification, return it
            if evaluation.score == "pass":
                return f"✅ VERIFIED: {content}"
            
            # Create an agent to improve the content
            improver_agent = Agent(
                name=f"content_improver_{verification_id}",
                instructions=f"You improve content based on feedback. Content type: {content_type}."
            )
            
            # Run the improver agent
            improved_result = await Runner.run(
                improver_agent,
                f"Original content: {content}\n\nFeedback: {evaluation.feedback}\n\nPlease improve this content based on the feedback."
            )
            
            # Return the improved content
            return f"🔄 IMPROVED: {improved_result.final_output}\n\nOriginal feedback: {evaluation.feedback}"
    
    def get_verify_function_tool(self):
        """Get the verification function tool"""
        
        # Define a wrapper with proper parameter annotation
        async def verify_content_wrapper(content: str, content_type: Literal["general", "research", "social_media", "linkedin", "instagram"]) -> str:
            """Verify and improve content"""
            return await self._verify_content_impl(content, content_type)
        
        # Use the function_tool decorator to create the tool
        verify_tool = function_tool(
            verify_content_wrapper,
            name_override="verify_content",
            description_override="Verify and improve content using the LLM as a Judge pattern. Use this tool for all content that needs verification."
        )
        
        return verify_tool