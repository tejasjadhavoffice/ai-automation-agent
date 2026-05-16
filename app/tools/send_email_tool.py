import logging
import re
import smtplib
from email.message import EmailMessage

from langchain_core.tools import tool

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Sends an email to the given recipient via SMTP."""
    settings = get_settings()

    if not settings.email_from.strip():
        logger.warning("step=send_email decision=misconfigured reason=EMAIL_FROM_not_set")
        return "Error: EMAIL_FROM is not configured in environment"
    
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", to):
        logger.warning("step=send_email decision=invalid_recipient to=%s", to)
        return f"Error: '{to}' is not a valid email address. Please provide a real recipient email."

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
        logger.error("step=send_email input=to=%s decision=smtp_error err=%s", to, exc, exc_info=True)
        return f"Error: send_email failed: {exc}"

    return f"Email sent successfully to {to}"
