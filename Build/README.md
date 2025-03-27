# Agents API Project

This project explores building applications with OpenAI's Agents API, focusing especially on multi-agent orchestration and patterns.

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp .env.example .env
   # Edit .env file to add your keys
   ```
4. Required API keys:
   - **OPENAI_API_KEY**: Your OpenAI API key for agent functionality
   - **REPLICATE_API_TOKEN** (optional): Only needed for image generation features
5. Run the project setup: `python cli.py setup`

### Troubleshooting

If you encounter an error like `openai.OpenAIError: The api_key client option must be set`, make sure:
1. You've copied `.env.example` to `.env`
2. You've added your actual OpenAI API key to the `.env` file
3. The API key is valid and has sufficient permissions

You can verify your API key configuration by running:
```bash
python test_api_key.py
```

For testing the verification system, you have several options:
```bash
# Original SDK-style LLM as a Judge example
python examples/sdk_verify_example.py

# Simplified verification test with direct integration
python examples/simple_verify.py
```

## Usage

The project includes a command-line interface for running examples:

```bash
# Run setup
python cli.py setup

# Show configuration
python cli.py config

# Run a specific example
python cli.py example <example_name>

# Run multi-agent examples
python cli.py multi-agent simple        # Simple orchestrator example
python cli.py multi-agent advanced      # Advanced orchestration & parallelization
python cli.py multi-agent verification  # Verification and improvement system
python cli.py multi-agent sdk-verify    # SDK-style verification example
```

## Multi-Agent Patterns

This project demonstrates several multi-agent patterns:

1. **Orchestrator Pattern**: A main agent coordinates specialized agents to complete tasks
2. **Hierarchical Agents**: Agents can contain and manage other specialized agents (e.g., social media → LinkedIn/Instagram)
3. **Parallelization Pattern**: Multiple agents work in parallel with results selected by an evaluator
4. **Agents-as-Tools Pattern**: Agents are exposed as tools to other agents
5. **Function Tools Pattern**: Python functions are used as tools for agents
6. **Conversation Memory**: Agents maintain context between conversation turns
7. **External API Integration**: Agents can interact with external services through function tools (e.g., image generation)
8. **LLM as a Judge Pattern**: An evaluator agent verifies and improves the output from other agents

## Tools

The project includes several types of tools for agents to use:

1. **Hosted Tools**:
   - WebSearchTool: Allows the research agent to search the web for up-to-date information

2. **Function Tools**:
   - `get_current_time()`: Returns the current date and time
   - `calculate_days_between()`: Calculates the number of days between two dates
   - `format_data()`: Formats data into JSON or text format
   - `generate_image()`: Creates images using the Replicate Flux-1.1-Pro AI model

## Features

### Verification System

The system implements the "LLM as a Judge" pattern to verify and improve content:

- **Iterative Improvement**: Content is verified and improved through multiple iterations
- **Quality Evaluation**: A dedicated verifier agent assesses content quality against specific criteria
- **Feedback Mechanism**: Concrete, actionable feedback is provided for each improvement cycle
- **Scoring System**: Content is rated as "pass", "needs_improvement", or "fail" 
- **Configurable Attempts**: Maximum verification attempts can be configured (default: 3)
- **Specialized Verification**: Different verification methods for research, social media, and images

The verification process ensures that all content meets high standards by:
1. Generating initial content with a specialized agent
2. Evaluating the content with a verifier agent
3. Getting feedback and a quality score
4. Implementing improvements based on feedback
5. Repeating until the content passes verification or reaches max attempts

### Image Generation

The system includes a graphic designer agent that can generate custom images:

- **AI-Powered Image Creation**: Generates images using the Replicate Flux-1.1-Pro model
- **Detailed Prompt Engineering**: Crafts optimal prompts for high-quality image generation
- **Customizable Aspect Ratios**: Supports different formats (1:1, 16:9, 9:16, 4:3, 3:4)
- **Social Media Integration**: Creates visuals that complement social media content
- **Visual Storytelling**: Designs images that reinforce the key messages in content
- **Verified Quality**: Images can be verified and improved by the verification system

Key capabilities:
- Creating standalone images from detailed descriptions
- Generating platform-optimized visuals for social media posts
- Producing images with appropriate style and tone for different contexts
- Supporting integrated content+image workflows with the social media agent

Example use cases:
- Generating product imagery for social media marketing
- Creating visual companions for LinkedIn articles
- Designing eye-catching visuals for Instagram posts
- Illustrating concepts mentioned in research findings

### Social Media Content Creation

The system includes a specialized social media agent structure:

- **Social Media Agent**: Orchestrates platform-specific content creation
  - **LinkedIn Agent**: Creates professional, business-focused content
  - **Instagram Agent**: Creates visual, lifestyle-oriented content

Key features:
- **Direct Content Delivery**: Social media content is delivered exactly as created by platform agents, without summarization or modification
- **Ready-to-Use Output**: All social media content is formatted to be directly copied and pasted to the respective platforms
- **Platform-Specific Optimization**: Content is tailored to each platform's unique characteristics and audience
- **Hierarchical Orchestration**: The main agent intelligently delegates to the appropriate platform agents

LinkedIn content includes:
- Professional tone and industry insights
- Attention-grabbing headlines
- Strategic hashtags and calls-to-action
- Content optimized for LinkedIn's character limits

Instagram content includes:
- Visually-oriented, engaging captions
- Appropriate emoji usage
- Strategic hashtag groups
- Content optimized for mobile consumption

Example use cases:
- Creating ready-to-post LinkedIn articles
- Crafting Instagram captions with trending hashtags
- Developing coordinated multi-platform campaigns

### Conversation Memory

The system maintains conversation history between turns, allowing agents to remember context from previous exchanges. This enables:

- Follow-up questions without repeating context
- Referring to previously mentioned locations or topics
- Building on previous responses

The memory is implemented by:
1. Storing the conversation history from each run
2. Appending new user queries to the history
3. Passing the full history to subsequent agent runs

### Modular Prompt Management

The system uses a dedicated prompt management solution:

- **External Prompt Files**: All agent prompts are stored in separate text files in the `/prompts` directory
- **Dynamic Loading**: Prompts are loaded at runtime using the `PromptLoader` utility
- **Caching Mechanism**: Prompts are cached after first load for better performance
- **Hot Reloading**: Supports forcing prompt reloads when prompt files are updated

Benefits:
- Easier maintenance and updates of agent prompts
- Clear separation of prompt content from code
- Ability to version control prompts independently
- Simplified collaboration on prompt engineering

### Structure

- `src/`: Source code for agent implementations
  - `multi_agent_manager.py`: Manager class for multi-agent orchestration
  - `utils.py`: Utility functions for configuration and environment
  - `tools.py`: Custom tool implementations for agents
  - `prompt_loader.py`: Utility for loading prompts from files
- `prompts/`: Modular prompt files for each agent
  - `research_agent.txt`: Instructions for the research agent
  - `linkedin_agent.txt`: Instructions for the LinkedIn content agent
  - `instagram_agent.txt`: Instructions for the Instagram content agent
  - `social_media_agent.txt`: Instructions for the social media orchestrator
  - `summarizer_agent.txt`: Instructions for the summarization agent
  - `graphic_designer_agent.txt`: Instructions for the graphic designer agent
  - `orchestrator_agent.txt`: Instructions for the main orchestrator agent
- `examples/`: Example applications using the agents
  - `multi_agent_example.py`: Simple multi-agent orchestration with memory
  - `advanced_multi_agent.py`: Advanced multi-agent patterns
- `tests/`: Test suite
- `config/`: Configuration files
- `cli.py`: Command-line interface for running examples

## Future Work

Upcoming features and improvements:

- More custom tools for real-world capabilities
- Custom guardrails for agent output validation
- Structured agent output for improved parsing
- Enhanced tracing and debugging capabilities
- Integration with external APIs and data sources
