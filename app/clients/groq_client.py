from langchain_groq import ChatGroq
from app.config.settings import Settings


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.llm = ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=settings.groq_temperature,
            max_retries=settings.groq_max_retries,
        )
