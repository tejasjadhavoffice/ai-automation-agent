def execute_summarise_text(arguments: dict) -> dict:
    text_value = arguments.get("text", "")
    if not isinstance(text_value, str) or not text_value.strip():
        return {"success": False, "message": "summarise_text requires a non-empty 'text' string", "data": {}}

    words = text_value.split()
    word_count = len(words)
    preview_words = words[:50]
    preview = " ".join(preview_words)
    if word_count > 50:
        preview += " ..."

    summary = (
        f"Summary preview ({min(word_count, 50)} of {word_count} words): {preview}"
    )
    return {
        "success": True,
        "message": "Text summarised successfully",
        "data": {"summary": summary, "word_count": word_count},
    }
