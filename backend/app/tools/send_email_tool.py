import smtplib
from email.message import EmailMessage

from app.config.settings import AppSettings


def execute_send_email(arguments: dict, settings: AppSettings) -> dict:
    recipient = arguments.get("to")
    subject = arguments.get("subject", "")
    body = arguments.get("body", "")

    if not isinstance(recipient, str) or not recipient.strip():
        return {"success": False, "message": "send_email requires a non-empty 'to' string", "data": {}}
    if not settings.email_from.strip():
        return {"success": False, "message": "EMAIL_FROM is not configured in environment", "data": {}}

    message = EmailMessage()
    message["From"] = settings.email_from
    message["To"] = recipient
    message["Subject"] = str(subject)
    message.set_content(str(body))

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as smtp:
            smtp.starttls()
            if settings.smtp_user and settings.smtp_password:
                smtp.login(settings.smtp_user, settings.smtp_password)
            smtp.send_message(message)
    except Exception as exc:
        return {
            "success": False,
            "message": f"send_email failed: {exc}",
            "data": {"to": recipient, "subject": str(subject)},
        }

    return {
        "success": True,
        "message": "Email sent successfully",
        "data": {"to": recipient, "subject": str(subject)},
    }
