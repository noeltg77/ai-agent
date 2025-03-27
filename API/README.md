# Multi-Agent API with Verification

This API provides access to a multi-agent system with verification capabilities using the "LLM as a Judge" pattern. It enables users to interact with a coordinated system of specialized agents through a REST API interface.

## Features

- **Multi-Agent Architecture**: Orchestrated system of specialized agents for research, social media content, image generation, and more
- **Verification System**: LLM as a Judge pattern for verifying and improving outputs
- **Session Management**: Maintains conversation context across multiple interactions
- **Detailed Tracing**: Full trace visibility for all operations, including verification steps
- **Docker Support**: Easy deployment with Docker and docker-compose

## API Endpoints

- **GET /**: Basic API information
- **GET /health**: Health check endpoint
- **GET /agents**: List available agents and their capabilities
- **POST /chat**: Main endpoint for interacting with the multi-agent system

## Deployment with Coolify

This API is designed to be easily deployed on Coolify.

### Deployment Steps

1. Fork this repository to your GitHub account
2. Create a new service in Coolify pointing to your forked repository
3. Set the following environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key with access to Assistant API capabilities
4. Configure the deployment to use the Dockerfile at `api/Dockerfile`
5. Deploy the service

## Local Development

### Prerequisites

- Python 3.10+
- OpenAI API key with Assistants API access

### Setup

1. Clone the repository
2. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-...
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running Locally

Start the API server:

```
uvicorn api.app:app --reload
```

The API will be available at http://localhost:8000.

### Using Docker

You can also run the API using Docker:

```
cd api
docker-compose up
```

## Example Usage

### Chat Endpoint

```bash
curl -X 'POST' \
  'http://localhost:8000/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": "Create a LinkedIn post about AI advancements in healthcare",
  "use_verification": true
}'
```

### Response Format

```json
{
  "session_id": "abc123de",
  "response": "LinkedIn Post: Revolutionizing Healthcare Through AI\n\nExcited to share how artificial intelligence is transforming patient care! Recent breakthroughs in medical imaging AI can now detect early-stage conditions that human eyes might miss. These tools aren't replacing doctors—they're empowering them.\n\nThe real game-changer? Personalized treatment plans generated through machine learning that analyze thousands of similar cases, leading to improved outcomes and reduced hospital stays.\n\nWhat healthcare+AI innovations are you most excited about? Share your thoughts below!\n\n#HealthcareInnovation #AIinMedicine #DigitalTransformation",
  "verification_details": [
    {
      "attempt": 1,
      "output": "...",
      "feedback": "...",
      "score": "pass"
    }
  ],
  "conversation_history": [
    {
      "role": "user",
      "content": "Create a LinkedIn post about AI advancements in healthcare"
    },
    {
      "role": "assistant",
      "content": "LinkedIn Post: Revolutionizing Healthcare Through AI\n\nExcited to share how artificial intelligence is transforming patient care! Recent breakthroughs in medical imaging AI can now detect early-stage conditions that human eyes might miss. These tools aren't replacing doctors—they're empowering them.\n\nThe real game-changer? Personalized treatment plans generated through machine learning that analyze thousands of similar cases, leading to improved outcomes and reduced hospital stays.\n\nWhat healthcare+AI innovations are you most excited about? Share your thoughts below!\n\n#HealthcareInnovation #AIinMedicine #DigitalTransformation"
    }
  ],
  "trace_id": "trace_abcdefg123456"
}