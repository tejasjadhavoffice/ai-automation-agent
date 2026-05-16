import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from app.config.settings import get_settings

MAX_SUMMARISE_CHARS = 6000

logger = logging.getLogger(__name__)

@tool
def summarise_text(text: str) -> str:
    """Summarises the given text into 3-5 clear sentences using AI."""
    if not text or not text.strip():
        logger.warning("step=summarise_text decision=empty_input")
        return "Error: no text provided to summarise"

    settings = get_settings()
    trimmed = text[:MAX_SUMMARISE_CHARS]

    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=settings.groq_temperature,
    )
    response = llm.invoke([
        SystemMessage(content="Summarise the following text in 3-5 clear sentences. Return only the summary."),
        HumanMessage(content=trimmed),
    ])
    return response.content

