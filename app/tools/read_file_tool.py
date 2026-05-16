import logging
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def read_file(path: str) -> str:
    """Reads a local file from disk and returns its content."""
    target = Path(path)
    if not target.exists():
        logger.warning("step=read_file input=%s decision=not_found", path)
        return f"Error: File not found: {path}"
    text = target.read_text(encoding="utf-8")
    logger.info("step=read_file input=%s decision=ok output=chars=%d", path, len(text))
    return text
