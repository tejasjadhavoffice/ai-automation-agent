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
