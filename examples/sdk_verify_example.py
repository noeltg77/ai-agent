"""
Verification Example using OpenAI Agents SDK pattern
based on the llm_as_a_judge.py example from the SDK documentation
"""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Literal
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Set OpenAI API key explicitly before importing any OpenAI-related packages
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    print(f"API Key set: {api_key[:5]}...{api_key[-4:]}")
else:
    print("No OpenAI API key found! Please add it to .env")
    sys.exit(1)

# Add the project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents import Agent, ItemHelpers, Runner, TResponseInputItem, trace

"""
This example shows the LLM as a judge pattern. The first agent generates an outline for a story.
The second agent judges the outline and provides feedback. We loop until the judge is satisfied
with the outline.
"""

# This is our content generator agent
content_generator = Agent(
    name="content_generator",
    instructions=(
        "You generate a very short paragraph based on the user's input."
        "If there is any feedback provided, use it to improve the paragraph."
    ),
)


@dataclass
class EvaluationFeedback:
    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]


# This is our evaluator/verifier agent
evaluator = Agent[None](
    name="evaluator",
    instructions=(
        "You evaluate content and decide if it's good enough."
        "If it's not good enough, you provide feedback on what needs to be improved."
        "Never give it a pass on the first try."
    ),
    output_type=EvaluationFeedback,
)


async def main() -> None:
    msg = input("What kind of content would you like to generate? ")
    input_items: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    latest_content: str | None = None

    # We'll run the entire workflow in a single trace
    with trace("LLM as a judge"):
        while True:
            content_result = await Runner.run(
                content_generator,
                input_items,
            )

            input_items = content_result.to_input_list()
            latest_content = ItemHelpers.text_message_outputs(content_result.new_items)
            print("Content generated")
            print(f"\nGenerated content: {latest_content}\n")

            evaluator_result = await Runner.run(evaluator, input_items)
            result: EvaluationFeedback = evaluator_result.final_output

            print(f"Evaluator score: {result.score}")
            print(f"Feedback: {result.feedback}\n")

            if result.score == "pass":
                print("Content is good enough, exiting.")
                break

            print("Re-running with feedback")

            input_items.append({"content": f"Feedback: {result.feedback}", "role": "user"})

    print(f"Final content: {latest_content}")


if __name__ == "__main__":
    asyncio.run(main())