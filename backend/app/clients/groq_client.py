import time

from groq import Groq

from app.config.settings import AppSettings


class RetryableApiError(Exception):
    """Raised for retryable API errors."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class GroqChatClient:
    """Handles Groq chat calls with retry and backoff."""

    model_name = "openai/gpt-oss-120b"
    temperature = 0.2
    top_p = 1.0
    request_timeout_seconds = 30.0
    max_retries = 3
    retry_base_delay_seconds = 1.0

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.client = Groq(api_key=settings.groq_api_key)

    def complete_chat(self, system_prompt: str, user_prompt: str) -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    timeout=self.request_timeout_seconds,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                if not self._is_retryable_exception(exc):
                    raise
                if attempt >= self.max_retries:
                    raise
                delay_seconds = self.retry_base_delay_seconds * (2 ** (attempt - 1))
                time.sleep(delay_seconds)

        raise RuntimeError("Unexpected retry loop termination")

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, RetryableApiError)):
            return True
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True
        if isinstance(status_code, int) and 500 <= status_code <= 599:
            return True
        message = str(exc).lower()
        return "timeout" in message or "timed out" in message
