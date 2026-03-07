"""Test the served vibe-check agent with a question that often fails the first vibe check.

Uses a terse, factual prompt so the agent's first reply tends to be short and impersonal,
causing the vibe checker to return VIBE:N and trigger a retry (friendlier response).
Run with: uv run test_served_graph_vibe.py (ensure langgraph dev is running on port 2024).
"""
from langgraph_sdk import get_sync_client


def main():
    client = get_sync_client(url="http://localhost:2024")
    # Question designed to elicit a short, factual reply that may fail the vibe check
    # (friendly/positive tone) on first attempt, so we see the agent -> vibe_check -> agent loop.
    input_message = (
        "What is photosynthesis? Answer in the driest, most technical one sentence possible. No warmth and friendliness. Just the facts"
    )
    for chunk in client.runs.stream(
        None,  # Threadless run
        "agent_with_vibe_check",  # Assistant id from langgraph.json (vibe-check agent)
        input={
            "messages": [
                {
                    "role": "human",
                    "content": input_message,
                }
            ]
        },
        stream_mode="updates",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")


if __name__ == "__main__":
    main()
