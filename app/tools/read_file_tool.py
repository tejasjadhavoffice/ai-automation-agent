"""Tool: read a local file and return its content."""

from pathlib import Path

from langchain_core.tools import tool


@tool
def read_file(path: str) -> str:
    """Reads a local file from disk and returns its content."""
    target = Path(path)
    if not target.exists():
        return f"Error: File not found: {path}"
    return target.read_text(encoding="utf-8")
