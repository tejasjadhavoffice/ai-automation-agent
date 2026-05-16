import logging

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def fetch_data(url: str) -> str:
    """Fetches the content of a URL via HTTP GET and returns the response body."""
    try:
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("step=fetch_data input=%s decision=http_error err=%s", url, exc)
        return f"Error: fetch failed: {exc}"
    body = response.text[:5000]
    logger.info("step=fetch_data input=%s decision=ok output=status=%s chars=%d", url, response.status_code, len(body))
    return body
