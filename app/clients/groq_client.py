"""
Why ChatGroq instead of raw groq.Groq?
  - Built-in retry with exponential backoff (no manual retry loop needed)
  - Native tool-calling support (no manual JSON parsing needed)
  - Industry-standard LangChain interface for agent frameworks
"""
import logging

from langchain_groq import ChatGroq
from app.config.settings import Settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Groq LLM via LangChain (single shared ``ChatGroq`` for the whole app)."""

    def __init__(self, settings: Settings) -> None:
        self.llm = ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=settings.groq_temperature,
            max_retries=settings.groq_max_retries,
        )
        logger.info("LLMClient ready — model=%s", settings.groq_model)
