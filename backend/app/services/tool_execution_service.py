from app.config.settings import AppSettings
from app.models.llm_tool_decision import LlmToolDecision
from app.tools.fetch_data_tool import execute_fetch_data
from app.tools.read_file_tool import execute_read_file
from app.tools.send_email_tool import execute_send_email
from app.tools.summarise_text_tool import execute_summarise_text


class ToolExecutionService:
    """Executes the selected tool and returns a response dict."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def execute_tool_by_name(self, decision: LlmToolDecision) -> dict:
        if decision.tool_name == "no_tool":
            return {"success": False, "message": f"No tool selected: {decision.reason}", "data": {}}

        if decision.tool_name == "read_file":
            return execute_read_file(decision.arguments)
        if decision.tool_name == "send_email":
            return execute_send_email(decision.arguments, self.settings)
        if decision.tool_name == "fetch_data":
            return execute_fetch_data(decision.arguments)
        if decision.tool_name == "summarise_text":
            return execute_summarise_text(decision.arguments)

        return {"success": False, "message": f"Unsupported tool: {decision.tool_name}", "data": {}}
