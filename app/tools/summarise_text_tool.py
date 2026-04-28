"""
summarise_text_tool.py

Calls the Groq LLM to produce a real AI summary of the given text.
The api_key is passed in from ToolExecutionService so this tool
does not need to read environment variables itself.
"""

from groq import Groq


def execute_summarise_text(arguments: dict, api_key: str) -> dict:
    """
    Summarise the text in arguments["text"] using the Groq LLM.

    Returns a standard result dict:
        {"success": bool, "message": str, "data": {...}}
    """
    text_value = arguments.get("text", "")

    # --- Input validation (guardrail) ---
    if not isinstance(text_value, str) or not text_value.strip():
        return {
            "success": False,
            "message": "summarise_text requires a non-empty 'text' string",
            "data": {},
        }

    if len(text_value) > 6000:
        # Trim to avoid blowing the context window
        text_value = text_value[:6000] + "\n[text trimmed to 6000 chars]"

    # --- Build a simple prompt ---
    system_prompt = (
        "You are a helpful assistant. "
        "Summarise the following text in 3-5 clear sentences. "
        "Return only the summary, no extra commentary."
    )
    user_prompt = f"Text to summarise:\n\n{text_value}"

    # --- Call Groq ---
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            timeout=30.0,
        )
        summary = response.choices[0].message.content or ""
    except Exception as exc:
        return {
            "success": False,
            "message": f"summarise_text LLM call failed: {exc}",
            "data": {},
        }

    return {
        "success": True,
        "message": "Text summarised successfully",
        "data": {
            "summary": summary,
            "original_word_count": len(text_value.split()),
        },
    }
