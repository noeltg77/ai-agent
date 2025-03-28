"""
FastAPI application for exposing the multi-agent system as a REST API.
"""
from __future__ import annotations

import os
import asyncio
from typing import Optional, Dict, Any, List, Union
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Load environment variables
load_dotenv()

# Set OpenAI API key explicitly for proper tracing
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

# Import after setting API key
from agents import Agent, Runner, trace, gen_trace_id
from src.prompt_loader import PromptLoader
from src.multi_agent_manager import MultiAgentManager
from src.verification_sdk import Verification

# Initialize the prompt loader
PromptLoader.initialize()

# Create the application
app = FastAPI(
    title="Multi-Agent API",
    description="API for interacting with a multi-agent system with verification capabilities",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global multi-agent manager for handling sessions
agent_manager = MultiAgentManager()

# Global verification system
verification = Verification(max_attempts=2)

# Pydantic models for API
class ConversationMessage(BaseModel):
    role: str = Field(..., description="The role of the message sender (user or assistant)")
    content: str = Field(..., description="The content of the message")

class SessionRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Session ID for continuing a conversation")
    input: str = Field(..., description="User input for the agent system")
    use_verification: bool = Field(True, description="Whether to use verification on the response")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the request")

class SessionResponse(BaseModel):
    session_id: str = Field(..., description="Session ID for continuing the conversation")
    response: str = Field(..., description="Agent system response")
    verification_details: Optional[List[Dict[str, Any]]] = Field(None, description="Verification details if verification was used")
    conversation_history: List[ConversationMessage] = Field(..., description="Complete conversation history")
    trace_id: Optional[str] = Field(None, description="Trace ID for debugging")

# Utility to get trace URL
def get_trace_url(trace_id: str) -> str:
    """Generate a URL for viewing the trace in the OpenAI platform."""
    return f"https://platform.openai.com/traces/{trace_id}"

@app.get("/")
async def root():
    """Root endpoint returning basic API information."""
    return {
        "message": "Multi-Agent API with Verification",
        "version": "1.0.0",
        "endpoints": {
            "/": "This documentation",
            "/chat": "Chat with the multi-agent system",
            "/health": "Health check endpoint",
            "/agents": "List available agents",
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "api_key_configured": bool(api_key)}

@app.get("/agents")
async def list_agents():
    """List all available agents in the system."""
    return {
        "agents": [
            {
                "name": "orchestrator",
                "description": "Coordinates all other agents and handles task delegation",
                "tools": ["research", "create_social_media_content", "create_image", "summarize", "verify_content"]
            },
            {
                "name": "research",
                "description": "Searches the web for factual information",
                "tools": ["web_search"]
            },
            {
                "name": "social_media",
                "description": "Creates optimized content for social media platforms",
                "tools": ["create_linkedin_content", "create_instagram_content"]
            },
            {
                "name": "graphic_designer",
                "description": "Generates custom images based on descriptions",
                "tools": ["generate_image"]
            },
            {
                "name": "verification",
                "description": "Verifies and improves content using LLM as a Judge pattern",
                "tools": []
            }
        ]
    }

@app.post("/chat", response_model=SessionResponse)
async def chat(request: SessionRequest, background_tasks: BackgroundTasks):
    """
    Chat with the multi-agent system.
    
    This endpoint handles:
    - Session management (continuing conversations)
    - Input processing
    - Optional verification of responses
    - Trace logging
    
    Returns the agent response and session details.
    """
    # Generate a session ID if not provided
    session_id = request.session_id or gen_trace_id()[:8]
    
    # Get or create the agent session
    session = agent_manager.get_or_create_session(session_id)
    
    # Create trace ID for this request
    trace_id = gen_trace_id()
    
    # Process the request with tracing
    with trace(f"API request - {session_id}", trace_id=trace_id):
        try:
            # Prepare input with conversation history if available
            if session.conversation_history:
                # Make sure we're not duplicating user messages
                # The session.conversation_history already contains previous turns in the format:
                # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
                # So we just need to add the new user message
                current_input = session.conversation_history.copy() + [{"role": "user", "content": request.input}]
                print(f"DEBUG: Using conversation history: {current_input}")
            else:
                # First message - just use the raw input as a string
                current_input = request.input
                print(f"DEBUG: First message, no history: {current_input}")
            
            # Process with or without verification based on request
            if request.use_verification:
                # Use direct verification
                final_output, verification_history = await verification.verify_and_improve(
                    session.orchestrator_agent,
                    current_input,
                    context=request.context,
                    trace_name=f"API verification - {session_id}"
                )
                
                # Update conversation history
                if session.conversation_history:
                    session.conversation_history = session.conversation_history + [
                        {"role": "user", "content": request.input},
                        {"role": "assistant", "content": final_output}
                    ]
                else:
                    session.conversation_history = [
                        {"role": "user", "content": request.input},
                        {"role": "assistant", "content": final_output}
                    ]
                
                # Convert to model format - handle complex message types from the SDK
                conversation_history = []
                for msg in session.conversation_history:
                    try:
                        # Handle regular messages with role and content
                        if "role" in msg and "content" in msg:
                            # If content is a string, use it directly
                            if isinstance(msg["content"], str):
                                conversation_history.append(
                                    ConversationMessage(role=msg["role"], content=msg["content"])
                                )
                            # If content is a list (complex message structures), try to extract text
                            elif isinstance(msg["content"], list):
                                # For assistant messages with complex content structure
                                extracted_content = ""
                                for content_item in msg["content"]:
                                    if isinstance(content_item, dict) and "text" in content_item:
                                        extracted_content += content_item["text"] + "\n"
                                if extracted_content:
                                    conversation_history.append(
                                        ConversationMessage(role=msg["role"], content=extracted_content.strip())
                                    )
                        # Handle function calls
                        elif "type" in msg and msg.get("type") == "function_call":
                            # Add function calls as system messages
                            call_description = f"Function call: {msg.get('name', 'unknown')}"
                            if "arguments" in msg:
                                call_description += f" with arguments: {msg['arguments']}"
                            conversation_history.append(
                                ConversationMessage(role="system", content=call_description)
                            )
                        # Handle function call outputs
                        elif "type" in msg and msg.get("type") == "function_call_output":
                            if "output" in msg:
                                conversation_history.append(
                                    ConversationMessage(role="system", content=f"Function output: {msg['output']}")
                                )
                    except Exception as e:
                        print(f"Warning: Error processing message for conversation history: {e}")
                        # Skip this message if it can't be processed
                
                return SessionResponse(
                    session_id=session_id,
                    response=final_output,
                    verification_details=verification_history,
                    conversation_history=conversation_history,
                    trace_id=trace_id
                )
            else:
                # For non-verification mode, use a simplified direct response approach
                # that bypasses potential version compatibility issues
                try:
                    # Create a simplified trace first
                    with trace(f"API request - {session_id}", trace_id=trace_id):
                        # This is a minimal implementation using direct run
                        # that avoids complex result structure handling
                        try:
                            # Try the new-style Runner call first - CRITICAL: pass the conversation history properly
                            # This is what ensures the agent maintains context between interactions
                            # Note: current_input is either a string (first message) or a list of chat messages
                            print(f"DEBUG: Running with input type: {type(current_input)}")
                            result = await Runner.run(
                                session.orchestrator_agent, 
                                current_input,  # This contains the conversation history + new user input
                                context=request.context
                            )
                            
                            # For debugging - check what the expected input format would be for next turn
                            if hasattr(result, 'to_input_list') and callable(result.to_input_list):
                                try:
                                    next_turn_input = result.to_input_list()
                                    print(f"DEBUG: Next turn input list would be: {next_turn_input}")
                                except Exception as e:
                                    print(f"DEBUG: Error getting next turn input list: {str(e)}")
                            
                            # Attempt to get the final output with graceful degradation
                            if hasattr(result, 'final_output'):
                                final_output = result.final_output
                            elif hasattr(result, 'output'):
                                final_output = result.output
                            elif hasattr(result, 'text'):
                                final_output = result.text
                            elif isinstance(result, str):
                                final_output = result
                            else:
                                # Very simple string conversion fallback
                                final_output = str(result)
                            
                        except Exception as runner_error:
                            # Log but continue with a simpler approach
                            print(f"Runner.run error: {str(runner_error)}. Using agent direct call.")
                            
                            # Ultra-simple fallback: call the agent directly
                            # This bypasses all complex result handling
                            # We still need to format the conversation history properly when using .generate
                            formatted_input = ""
                            if isinstance(current_input, list):
                                # Format conversation history for direct call
                                formatted_input = "\n\n".join([
                                    f"{msg['role'].upper()}: {msg['content']}" 
                                    for msg in current_input
                                ])
                            else:
                                formatted_input = current_input
                            
                            response = await session.orchestrator_agent.generate(formatted_input)
                            final_output = response.text
                
                    # Debug log to see what's happening with the conversation history
                    print(f"Session {session_id} conversation history before update: {session.conversation_history}")
                    
                    # Following the SDK pattern for conversation history
                    try:
                        # If we got a result from Runner.run, use the proper method to get the conversation history
                        if hasattr(result, 'to_input_list') and callable(result.to_input_list):
                            try:
                                # This is the recommended way to get the conversation history from the SDK
                                # It properly formats all the turns, tools calls, etc.
                                session.conversation_history = result.to_input_list()
                                print(f"DEBUG: Updated conversation history using result.to_input_list()")
                            except Exception as e:
                                print(f"DEBUG: Error updating conversation history from result: {str(e)}. Using fallback.")
                                # Fallback to manual update
                                if session.conversation_history:
                                    session.conversation_history = session.conversation_history + [
                                        {"role": "user", "content": request.input},
                                        {"role": "assistant", "content": final_output}
                                    ]
                                else:
                                    session.conversation_history = [
                                        {"role": "user", "content": request.input},
                                        {"role": "assistant", "content": final_output}
                                    ]
                        else:
                            # Manual update if no result.to_input_list method
                            if session.conversation_history:
                                session.conversation_history = session.conversation_history + [
                                    {"role": "user", "content": request.input},
                                    {"role": "assistant", "content": final_output}
                                ]
                            else:
                                session.conversation_history = [
                                    {"role": "user", "content": request.input},
                                    {"role": "assistant", "content": final_output}
                                ]
                    except Exception as e:
                        print(f"DEBUG: General error updating conversation history: {str(e)}. Using minimal fallback.")
                        # Ultimate fallback
                        if session.conversation_history:
                            session.conversation_history = session.conversation_history + [
                                {"role": "user", "content": request.input},
                                {"role": "assistant", "content": final_output}
                            ]
                        else:
                            session.conversation_history = [
                                {"role": "user", "content": request.input},
                                {"role": "assistant", "content": final_output}
                            ]
                    
                    # Debug log after update
                    print(f"Session {session_id} conversation history after update: {session.conversation_history}")
                    
                    # Convert to model format - handle complex message types from the SDK
                    conversation_history = []
                    for msg in session.conversation_history:
                        try:
                            # Only include messages that have both role and content as simple fields
                            if "role" in msg and "content" in msg:
                                # If content is a string, use it directly
                                if isinstance(msg["content"], str):
                                    conversation_history.append(
                                        ConversationMessage(role=msg["role"], content=msg["content"])
                                    )
                                # If content is a list (complex message structures), try to extract text
                                elif isinstance(msg["content"], list):
                                    # For assistant messages with complex content structure
                                    extracted_content = ""
                                    for content_item in msg["content"]:
                                        if isinstance(content_item, dict) and "text" in content_item:
                                            extracted_content += content_item["text"] + "\n"
                                    if extracted_content:
                                        conversation_history.append(
                                            ConversationMessage(role=msg["role"], content=extracted_content.strip())
                                        )
                        except Exception as e:
                            print(f"Warning: Error processing message for conversation history: {e}")
                            # Skip this message if it can't be processed
                    
                    # Return the response
                    return SessionResponse(
                        session_id=session_id,
                        response=final_output,
                        verification_details=None,
                        conversation_history=conversation_history,
                        trace_id=trace_id
                    )
                    
                except Exception as e:
                    # Log any errors that occurred during the simplified approach
                    print(f"Error in simplified non-verification mode: {str(e)}")
                    
                    # Ultimate fallback - just return what we have
                    # This ensures we never fail with 500 for this path
                    if not 'final_output' in locals():
                        final_output = f"I apologize, but I encountered an error processing your request. Details: {str(e)}"
                    
                    if session.conversation_history:
                        conversation_history = [
                            ConversationMessage(role=msg["role"], content=msg["content"])
                            for msg in session.conversation_history + [
                                {"role": "user", "content": request.input},
                                {"role": "assistant", "content": final_output}
                            ]
                        ]
                    else:
                        conversation_history = [
                            ConversationMessage(role="user", content=request.input),
                            ConversationMessage(role="assistant", content=final_output)
                        ]
                    
                    # Always return a valid response, never fail with 500
                    return SessionResponse(
                        session_id=session_id,
                        response=final_output,
                        verification_details=None,
                        conversation_history=conversation_history,
                        trace_id=trace_id
                    )
                
        except Exception as e:
            # Log the error in the background to avoid blocking the response
            background_tasks.add_task(print, f"Error processing request: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Run the application if executed directly
if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    # Run with uvicorn
    uvicorn.run("API.app:app", host="0.0.0.0", port=port, reload=True)