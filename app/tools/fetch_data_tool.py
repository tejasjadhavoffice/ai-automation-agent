"""Tool: fetch data from a URL via HTTP GET."""

import httpx
from langchain_core.tools import tool


@tool
def fetch_data(url: str) -> str:
    """Fetches data from a URL via HTTP GET and returns the response body."""
    try:
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return f"Error: fetch failed: {exc}"
    return response.text[:5000]
