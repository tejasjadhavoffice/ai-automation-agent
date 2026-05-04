"""Tool: send an email via SMTP."""

import smtplib
from email.message import EmailMessage

from langchain_core.tools import tool

from app.config.settings import get_settings


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Sends an email to the given recipient via SMTP."""
    settings = get_settings()

    if not settings.email_from.strip():
        return "Error: EMAIL_FROM is not configured in environment"

    message = EmailMessage()
    message["From"] = settings.email_from
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as smtp:
            smtp.starttls()
            if settings.smtp_user and settings.smtp_password:
                smtp.login(settings.smtp_user, settings.smtp_password)
            smtp.send_message(message)
    except Exception as exc:
        return f"Error: send_email failed: {exc}"

    return f"Email sent successfully to {to}"
