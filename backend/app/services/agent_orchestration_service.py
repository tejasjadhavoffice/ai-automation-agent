import json
import re

from pydantic import ValidationError

from app.clients.groq_client import GroqChatClient
from app.models.llm_tool_decision import LlmToolDecision
from app.prompts.prompt_factory import SYSTEM_PROMPT, build_user_prompt
from app.services.tool_execution_service import ToolExecutionService

class AgentOrchestrationService:
    """Coordinates prompt creation, model call, parsing, and tool execution."""

    def __init__(
        self,
        groq_client: GroqChatClient,
        tool_execution_service: ToolExecutionService,
    ) -> None:
        self.groq_client = groq_client
        self.tool_execution_service = tool_execution_service

    def run_once(self, user_request: str, prompt_style: str) -> dict:
        user_prompt = build_user_prompt(user_request=user_request, style=prompt_style)
        llm_response_text = self.groq_client.complete_chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        decision = self.parse_tool_decision_json(llm_response_text)
        tool_result = self.tool_execution_service.execute_tool_by_name(decision)

        print(
            f"[agent_step] style={prompt_style} selected_tool={decision.tool_name} "
            f"tool_success={tool_result.get('success')}"
        )

        return {
            "prompt_style": prompt_style,
            "llm_response_text": llm_response_text,
            "selected_tool": decision.tool_name,
            "tool_result": tool_result,
        }

    @staticmethod
    def format_output(result: dict) -> str:
        return json.dumps(result, indent=2)

    @staticmethod
    def parse_tool_decision_json(response_text: str) -> LlmToolDecision:
        cleaned_text = response_text.strip()
        json_payload = AgentOrchestrationService._extract_json(cleaned_text)
        try:
            parsed_dict = json.loads(json_payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from model: {exc}") from exc

        try:
            return LlmToolDecision.model_validate(parsed_dict)
        except ValidationError as exc:
            raise ValueError(f"JSON schema validation failed: {exc}") from exc

    @staticmethod
    def _extract_json(response_text: str) -> str:
        if response_text.startswith("{") and response_text.endswith("}"):
            return response_text

        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            return match.group(0)
        raise ValueError("No JSON object found in model response")
