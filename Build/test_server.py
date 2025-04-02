"""
Simple test server to check connectivity.
"""
import os
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Create a simple app
app = FastAPI(title="Test API Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Test API Server is running",
        "status": "healthy",
        "timestamp": time.time()
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }

@app.get("/info")
async def info(request: Request):
    """Returns detailed information about the request."""
    return {
        "client": request.client.host,
        "headers": dict(request.headers),
        "query_params": dict(request.query_params),
        "timestamp": time.time(),
        "server_port": os.getenv("PORT", "8000")
    }

@app.post("/echo")
async def echo(request: Request):
    """Echo back the request body."""
    try:
        body = await request.json()
        return {
            "message": "Echo successful",
            "received": body,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "message": f"Error parsing request: {str(e)}",
            "timestamp": time.time()
        }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting test server on port {port}")
    uvicorn.run("test_server:app", host="0.0.0.0", port=port)