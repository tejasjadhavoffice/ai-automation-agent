import json
import logging
import re

from pydantic import ValidationError

from app.clients.groq_client import GroqChatClient
from app.models.llm_tool_decision import LlmToolDecision, ReactStep

from app.prompts.prompt_factory import (
    REACT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_react_user_message,
    build_user_prompt,
)
from app.services.tool_execution_service import ToolExecutionService


class AgentOrchestrationService:
    max_steps = 5
    max_memory_chars = 4500
    stuck_repeat_limit = 2

    def __init__(
        self,
        groq_client: GroqChatClient,
        tool_execution_service: ToolExecutionService,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.groq = groq_client
        self.tools = tool_execution_service

    def run_once(self, user_request: str, prompt_style: str) -> dict:
        """Week 1: single LLM call + one tool."""
        self.logger.info("run_once started")
        user_prompt = build_user_prompt(user_request=user_request, style=prompt_style)
        llm_text = self.groq.complete_chat(SYSTEM_PROMPT, user_prompt)
        decision = self._parse_json(llm_text, LlmToolDecision)
        self.logger.debug("run_once selected tool=%s", decision.tool_name)
        tool_result = self.tools.execute_tool_by_name(decision)
        return {
            "mode": "once",
            "prompt_style": prompt_style,
            "llm_response_text": llm_text,
            "selected_tool": decision.tool_name,
            "tool_result": tool_result,
        }

    def run_react(self, user_request: str, prompt_style: str) -> dict:
        """Week 2: ReAct loop with short-term memory, trim, stuck guard."""
        self.logger.info("run_react started")
        memory_lines: list[str] = []
        step_log: list[dict] = []
        last_sig: str | None = None
        repeats = 0

        for step in range(1, self.max_steps + 1):
            self.logger.info("STEP %s/%s", step, self.max_steps)
            memory_text = self._trim_memory(memory_lines)
            user_msg = build_react_user_message(user_request, prompt_style, memory_text)
            llm_text = self.groq.complete_chat(REACT_SYSTEM_PROMPT, user_msg)
            react = self._parse_json(llm_text, ReactStep)

            self.logger.info("THOUGHT: %s", react.thought)
            if react.subtasks:
                self.logger.info("SUBTASKS: %s", react.subtasks)

            if react.done:
                self.logger.info("DONE signaled by model")
                step_log.append(
                    {
                        "step": step,
                        "thought": react.thought,
                        "done": True,
                        "final_reason": react.reason,
                    }
                )
                return {
                    "mode": "react",
                    "prompt_style": prompt_style,
                    "stopped": "done",
                    "steps": step_log,
                    "final_message": react.reason or react.thought,
                }

            decision = LlmToolDecision(
                tool_name=react.tool_name,
                arguments=react.arguments,
                reason=react.reason,
            )
            sig = f"{react.tool_name}:{json.dumps(react.arguments, sort_keys=True)}"
            if sig == last_sig:
                repeats += 1
            else:
                repeats = 0
                last_sig = sig
            if repeats >= self.stuck_repeat_limit:
                self.logger.warning("STUCK: same tool+arguments repeated")
                step_log.append({"step": step, "thought": react.thought, "stuck": True})
                return {
                    "mode": "react",
                    "prompt_style": prompt_style,
                    "stopped": "stuck",
                    "steps": step_log,
                    "final_message": "Stuck: repeated identical action without progress.",
                }

            self.logger.info("ACT: tool=%s args=%s", react.tool_name, react.arguments)
            tool_result = self.tools.execute_tool_by_name(decision)
            obs_short = json.dumps(tool_result, default=str)[:800]
            self.logger.info("OBSERVE: %s", obs_short)

            step_log.append(
                {
                    "step": step,
                    "thought": react.thought,
                    "tool_name": react.tool_name,
                    "arguments": react.arguments,
                    "observation": tool_result,
                }
            )
            # Store the observation data so the LLM can use it in the next step.
            # We include the actual data payload (truncated) so the agent can
            # pass file content, API responses, etc. to the next tool.
            obs_data = json.dumps(tool_result.get("data", {}), default=str)[:1200]
            memory_lines.append(
                f"Step {step} | tool: {react.tool_name} | ok: {tool_result.get('success')} "
                f"| thought: {react.thought[:150]} "
                f"| observation_data: {obs_data}"
            )

        self.logger.warning("MAX STEPS reached: %s", self.max_steps)
        return {
            "mode": "react",
            "prompt_style": prompt_style,
            "stopped": "max_steps",
            "steps": step_log,
            "final_message": "Stopped after max steps.",
        }

    def _trim_memory(self, lines: list[str]) -> str:
        text = "\n".join(lines)
        if len(text) <= self.max_memory_chars:
            return text
        return text[-self.max_memory_chars :]

    @staticmethod
    def format_output(result: dict) -> str:
        return json.dumps(result, indent=2, default=str)

    @staticmethod
    def _parse_json(text: str, model_class: type):
        raw = text.strip()
        if raw.startswith("{") and raw.endswith("}"):
            payload = raw
        else:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                raise ValueError("No JSON object found in model response")
            payload = m.group(0)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from model: {exc}") from exc
        try:
            return model_class.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"JSON schema validation failed: {exc}") from exc
