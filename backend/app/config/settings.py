from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Centralized environment-backed configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    groq_api_key: str = Field(..., alias="GROQ_API_KEY")

    smtp_host: str = Field("", alias="SMTP_HOST")
    smtp_port: int = Field(587, alias="SMTP_PORT")
    smtp_user: str = Field("", alias="SMTP_USER")
    smtp_password: str = Field("", alias="SMTP_PASSWORD")
    email_from: str = Field("", alias="EMAIL_FROM")


@lru_cache
def get_settings() -> AppSettings:
    """Returns cached settings for the process lifetime."""

    return AppSettings()
