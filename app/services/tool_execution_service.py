"""
tool_execution_service.py

Holds the list of all agent tools in one place.
LangGraph uses this list to know which tools the agent can call.
"""

import logging

from app.tools.fetch_data_tool import fetch_data
from app.tools.read_file_tool import read_file
from app.tools.send_email_tool import send_email
from app.tools.summarise_text_tool import summarise_text

logger = logging.getLogger(__name__)


class ToolRegistryService:
    """
    Registry of all available tools.

    Why a class instead of a plain list?
      - OOP: easy to extend with new tools later (just add to the list)
      - Encapsulation: the rest of the code calls get_tools() — it doesn't
        need to know which tools exist or how they are imported
    """

    def __init__(self) -> None:
        self._tools = [read_file, fetch_data, summarise_text, send_email]
        logger.info("ToolRegistryService loaded %d tools", len(self._tools))

    def get_registered_tools(self) -> list:
        """Return the list of LangChain tool objects."""
        return self._tools
