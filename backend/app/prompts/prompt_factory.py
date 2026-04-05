from textwrap import dedent

SYSTEM_PROMPT = dedent(
    """
    You are a tool-router assistant.
    Choose exactly one tool that best matches the request.
    Return only JSON in this exact shape:
    {"tool_name":"...","arguments":{...},"reason":"..."}

    Allowed tool_name values:
    - read_file
    - send_email
    - fetch_data
    - summarise_text
    - no_tool

    Rules:
    - If no tool fits safely, use "no_tool" and explain in "reason".
    - Keep arguments minimal and valid for the chosen tool.
    - Never return markdown, code fences, or extra text.
    """
).strip()

REACT_SYSTEM_PROMPT = dedent(
    """
    You are a ReAct agent: Reason, Act with one tool, then observe results in the next message.

    Return ONLY JSON (no markdown) in this exact shape:
    {
      "thought": "why you choose this step",
      "subtasks": ["optional","high-level","steps","you","decomposed"],
      "done": false,
      "tool_name": "read_file|send_email|fetch_data|summarise_text|no_tool",
      "arguments": {},
      "reason": ""
    }

    Rules:
    - Break complex goals into subtasks in "subtasks" (can be empty after first plan).
    - Set "done": true when the user's goal is fully satisfied; then use tool_name "no_tool"
      and put a short final summary in "reason".
    - One tool per response. After tools run, you will see observations — plan the next tool.
    - If you cannot proceed, use no_tool with explanation in "reason".
    """
).strip()

FEW_SHOT_EXAMPLE = dedent(
    """
    Example:
    User: "Read file notes/today.txt"
    Assistant:
    {"tool_name":"read_file","arguments":{"path":"notes/today.txt"},"reason":""}
    """
).strip()

COT_HINT = dedent(
    """
    Think briefly about intent, then return only final JSON.
    """
).strip()


def build_user_prompt(user_request: str, style: str) -> str:
    if style == "few-shot":
        return f"{FEW_SHOT_EXAMPLE}\n\nUser: {user_request}"
    if style == "cot":
        return f"{COT_HINT}\n\nUser: {user_request}"
    return f"User: {user_request}"


def build_react_user_message(goal: str, style: str, memory_text: str) -> str:
    base = build_user_prompt(user_request=goal, style=style)
    if not memory_text.strip():
        return base
    return f"{base}\n\n--- Prior steps (short-term memory) ---\n{memory_text}"
