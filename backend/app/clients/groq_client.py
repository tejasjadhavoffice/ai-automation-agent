import time

from groq import Groq

from app.config.settings import AppSettings


class GroqChatClient:
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
                r = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    timeout=self.request_timeout_seconds,
                )
                return r.choices[0].message.content or ""
            except Exception as exc:
                if not self._retryable(exc) or attempt >= self.max_retries:
                    raise
                delay = self.retry_base_delay_seconds * (2 ** (attempt - 1))
                print(
                    f"[groq_retry] attempt {attempt}/{self.max_retries} "
                    f"sleep {delay:.1f}s err={type(exc).__name__}"
                )
                time.sleep(delay)
        raise RuntimeError("retry loop ended unexpectedly")

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        code = getattr(exc, "status_code", None)
        if code == 429:
            return True
        if isinstance(code, int) and 500 <= code <= 599:
            return True
        msg = str(exc).lower()
        return "timeout" in msg or "timed out" in msg
