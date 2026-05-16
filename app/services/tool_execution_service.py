from app.tools.fetch_data_tool import fetch_data
from app.tools.read_file_tool import read_file
from app.tools.send_email_tool import send_email
from app.tools.summarise_text_tool import summarise_text


class ToolRegistryService:

    def __init__(self) -> None:
        self._tools = [read_file, fetch_data, summarise_text, send_email]

    def get_registered_tools(self) -> list:
        return self._tools
