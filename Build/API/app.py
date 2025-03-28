"""
FastAPI application for exposing the multi-agent system as a REST API.
"""
from __future__ import annotations

import os
import asyncio
from typing import Optional, Dict, Any, List, Union
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks, Security, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import uvicorn
import secrets

# Load environment variables
load_dotenv()

# Set OpenAI API key explicitly for proper tracing
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

# API key authentication
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    # Generate a random API key if not set in environment
    API_KEY = secrets.token_urlsafe(32)
    print(f"Warning: API_KEY not found in environment. Generated random key: {API_KEY}")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=401,
        detail="Invalid API Key",
        headers={"WWW-Authenticate": "ApiKey"},
    )

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
            "/chat": "Chat with the multi-agent system (requires API key)",
            "/health": "Health check endpoint",
            "/agents": "List available agents (requires API key)",
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "api_key_configured": bool(api_key)}

@app.get("/agents")
async def list_agents(api_key: str = Depends(get_api_key)):
    """List all available agents in the system. Requires API key authentication."""
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
async def chat(request: SessionRequest, background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)):
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
    
    try:
        # Prepare input with conversation history if available
        if session.conversation_history:
            current_input = session.conversation_history + [{"role": "user", "content": request.input}]
        else:
            current_input = request.input
        
        # Process with or without verification based on request
        if request.use_verification:
            # Use direct verification with its own trace
            final_output, verification_history = await verification.verify_and_improve(
                session.orchestrator_agent,
                current_input,
                context=request.context,
                trace_name=f"API request - {session_id}",
                trace_id=trace_id
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
            
            # Convert to Pydantic model format
            conversation_history = [
                ConversationMessage(role=msg["role"], content=msg["content"])
                for msg in session.conversation_history
            ]
            
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
                        result = await Runner.run(
                            session.orchestrator_agent, 
                            current_input,  # This contains the conversation history + new user input
                            context=request.context
                        )
                        
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
                
                # Super simple conversation history update
                if session.conversation_history:
                    new_conversation = session.conversation_history + [
                        {"role": "user", "content": request.input},
                        {"role": "assistant", "content": final_output}
                    ]
                else:
                    new_conversation = [
                        {"role": "user", "content": request.input},
                        {"role": "assistant", "content": final_output}
                    ]
                
                # Save the new conversation history
                session.conversation_history = new_conversation
                
                # Debug log after update
                print(f"Session {session_id} conversation history after update: {session.conversation_history}")
                
                # Convert to model format
                conversation_history = [
                    ConversationMessage(role=msg["role"], content=msg["content"])
                    for msg in new_conversation
                ]
                
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