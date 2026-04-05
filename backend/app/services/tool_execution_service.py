from app.config.settings import AppSettings
from app.models.llm_tool_decision import LlmToolDecision
from app.tools.fetch_data_tool import execute_fetch_data
from app.tools.read_file_tool import execute_read_file
from app.tools.send_email_tool import execute_send_email
from app.tools.summarise_text_tool import execute_summarise_text


class ToolExecutionService:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._handlers = {
            "read_file": execute_read_file,
            "fetch_data": execute_fetch_data,
            "summarise_text": execute_summarise_text,
            "send_email": lambda a: execute_send_email(a, self._settings),
        }

    def execute_tool_by_name(self, decision: LlmToolDecision) -> dict:
        name = decision.tool_name
        if name == "no_tool":
            return {"success": False, "message": f"No tool selected: {decision.reason}", "data": {}}

        fn = self._handlers.get(name)
        if not fn:
            return {"success": False, "message": f"Unsupported tool: {name}", "data": {}}
        return fn(decision.arguments)
