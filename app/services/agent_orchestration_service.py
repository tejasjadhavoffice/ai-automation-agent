import json
import logging
import re
import uuid

from pydantic import ValidationError

from app.clients.groq_client import GroqChatClient
from app.guardrails.guardrail_checker import GuardrailChecker
from app.logging_setup import log_step
from app.models.llm_tool_decision import LlmToolDecision, ReactStep

from app.prompts.prompt_factory import (
    REACT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    WEEK4_REACT_SYSTEM_PROMPT,
    build_react_user_message,
    build_user_prompt,
    build_week4_user_message,
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

    def run_week4(self, user_request: str) -> dict:
        """
        Week 4: Production ReAct loop.

        New features vs run_react():
          1. trace_id — short ID stamped on every log line for this run.
          2. Ambiguity check — if LLM sets needs_clarification=True, stop and
             print the question. User must rerun with a clearer request.
          3. Guardrail at every output→action boundary — validates tool_name
             and required arguments BEFORE executing any tool.
        """
        trace_id = uuid.uuid4().hex[:6]
        self.logger.info("[%s] Week4 run started | request: %s", trace_id, user_request[:80])
        log_step("Week4Agent", "start", user_request, "", "started", trace_id=trace_id)

        checker = GuardrailChecker()
        memory_lines: list[str] = []
        step_log: list[dict] = []
        last_sig: str | None = None
        repeats = 0

        for step in range(1, self.max_steps + 1):
            self.logger.info("[%s] STEP %s/%s", trace_id, step, self.max_steps)

            # Build prompt with short-term memory
            memory_text = self._trim_memory(memory_lines)
            user_msg = build_week4_user_message(user_request, memory_text, trace_id)

            # Call LLM — Week 4 uses the smarter 70b model for better instruction-following
            llm_text = self.groq.complete_chat(WEEK4_REACT_SYSTEM_PROMPT, user_msg, use_week4_model=True)
            react = self._parse_json(llm_text, ReactStep)

            self.logger.info("[%s] THOUGHT: %s", trace_id, react.thought)
            log_step("Week4Agent", f"step_{step}_thought", user_request[:100], react.thought, "thinking", trace_id=trace_id)

            # ── Ambiguity check (Week 4 new feature) ──────────────────────────
            if react.needs_clarification:
                self.logger.info("[%s] CLARIFICATION NEEDED: %s", trace_id, react.clarifying_question)
                log_step("Week4Agent", "clarification", user_request[:100], react.clarifying_question, "needs_clarification", trace_id=trace_id)
                return {
                    "trace_id": trace_id,
                    "stopped": "needs_clarification",
                    "clarifying_question": react.clarifying_question,
                    "message": f"Agent needs more info: {react.clarifying_question}",
                }

            # ── Done ──────────────────────────────────────────────────────────
            if react.done:
                self.logger.info("[%s] DONE", trace_id)
                log_step("Week4Agent", "done", "", react.reason, "done", trace_id=trace_id)
                step_log.append({"step": step, "thought": react.thought, "done": True, "final_reason": react.reason})
                return {
                    "trace_id": trace_id,
                    "stopped": "done",
                    "steps": step_log,
                    "final_message": react.reason or react.thought,
                }

            # ── Guardrail at output→action boundary (Week 4 new feature) ──────
            ok, guard_msg = checker.check_tool_decision(react.tool_name, react.arguments)
            log_step("Week4Agent", f"step_{step}_guardrail", react.tool_name, str(ok), guard_msg, trace_id=trace_id)
            if not ok:
                self.logger.warning("[%s] GUARDRAIL BLOCKED: %s", trace_id, guard_msg)
                step_log.append({"step": step, "thought": react.thought, "guardrail_blocked": guard_msg})
                return {
                    "trace_id": trace_id,
                    "stopped": "guardrail_blocked",
                    "steps": step_log,
                    "final_message": f"Guardrail blocked: {guard_msg}",
                }

            # ── Stuck guard ───────────────────────────────────────────────────
            sig = f"{react.tool_name}:{json.dumps(react.arguments, sort_keys=True)}"
            if sig == last_sig:
                repeats += 1
            else:
                repeats = 0
                last_sig = sig
            if repeats >= self.stuck_repeat_limit:
                self.logger.warning("[%s] STUCK", trace_id)
                log_step("Week4Agent", "stuck", sig, "", "stuck", trace_id=trace_id)
                return {"trace_id": trace_id, "stopped": "stuck", "steps": step_log, "final_message": "Stuck — repeated identical action."}

            # ── Execute tool ──────────────────────────────────────────────────
            decision = LlmToolDecision(tool_name=react.tool_name, arguments=react.arguments, reason=react.reason)
            self.logger.info("[%s] ACT: tool=%s", trace_id, react.tool_name)
            tool_result = self.tools.execute_tool_by_name(decision)

            obs_short = json.dumps(tool_result, default=str)[:800]
            self.logger.info("[%s] OBSERVE: %s", trace_id, obs_short[:200])
            log_step("Week4Agent", f"step_{step}_tool_{react.tool_name}", str(react.arguments)[:200], obs_short[:300], str(tool_result.get("success")), trace_id=trace_id)

            step_log.append({"step": step, "thought": react.thought, "tool_name": react.tool_name, "arguments": react.arguments, "observation": tool_result})

            obs_data = json.dumps(tool_result.get("data", {}), default=str)[:1200]
            memory_lines.append(
                f"Step {step} | tool: {react.tool_name} | ok: {tool_result.get('success')} "
                f"| thought: {react.thought[:150]} | data: {obs_data}"
            )

        self.logger.warning("[%s] MAX STEPS reached", trace_id)
        log_step("Week4Agent", "max_steps", "", "", "max_steps_reached", trace_id=trace_id)
        return {"trace_id": trace_id, "stopped": "max_steps", "steps": step_log, "final_message": "Stopped after max steps."}

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
