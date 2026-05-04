"""
prompt_factory.py

All system prompts and user-prompt builders in one place.
Easy to update prompts without touching business logic.
"""

from textwrap import dedent

# ── Week 1: single tool selection ─────────────────────────────────────────────

SINGLE_TOOL_SYSTEM_PROMPT = dedent("""
    You are a tool-router assistant.
    Choose exactly one tool that best matches the user request.

    Available tools: read_file, fetch_data, summarise_text, send_email.

    Rules:
    - If the request is unclear, ask a clarifying question instead of guessing.
    - Keep arguments minimal and valid for the chosen tool.
""").strip()

FEW_SHOT_EXAMPLE = dedent("""
    Example:
    User: "Read file notes/today.txt"
    You called: read_file(path="notes/today.txt")
""").strip()

COT_HINT = "Think briefly about the user's intent, then pick the right tool."


# ── Week 2/4: ReAct agent system prompt ───────────────────────────────────────

REACT_SYSTEM_PROMPT = dedent("""
    You are a ReAct automation agent. You reason step-by-step, use tools to
    gather information, and complete the user's goal.

    Available tools: read_file, fetch_data, summarise_text, send_email.

    Rules:
    - Break complex goals into smaller steps.
    - Use one tool per step. Wait for the tool result before calling the next tool.
    - When the goal is fully satisfied, respond with a final summary.
    - If the user's request is vague or missing details (no file path, no URL),
      ask a clarifying question instead of guessing.

    CRITICAL: When you call summarise_text, you MUST pass the FULL ACTUAL text
    content that was returned by a previous tool call. Never pass a description
    like "content from file" — copy the real text word-for-word.
""").strip()


def build_styled_user_prompt(user_request: str, style: str = "zero-shot") -> str:
    """
    Build the user message with optional prompt-engineering style.

    Styles (Week 1 concept):
      - zero-shot: just the request (simple, works for clear tasks)
      - few-shot:  adds an example (helps LLM follow the exact format)
      - cot:       adds a thinking hint (helps with complex/ambiguous tasks)
    """
    if style == "few-shot":
        return f"{FEW_SHOT_EXAMPLE}\n\nUser: {user_request}"
    if style == "cot":
        return f"{COT_HINT}\n\nUser: {user_request}"
    return f"User: {user_request}"
