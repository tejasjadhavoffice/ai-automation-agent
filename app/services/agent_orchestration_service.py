"""
agent_orchestration_service.py — The brain of the agent.

Two modes:
  run_once()  → Week 1: single LLM call + one tool (uses bind_tools)
  run_react() → Week 2/4: multi-step ReAct loop (uses LangGraph + MemorySaver)

Why LangGraph instead of a manual while-loop?
  - create_react_agent() implements the full ReAct pattern in ~5 lines
  - MemorySaver gives short-term memory across steps automatically
  - Built-in recursion_limit replaces our manual max_steps guard
  - Tool dispatch is automatic — no manual dictionary needed
"""

import json
import logging
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.guardrails.guardrail_checker import GuardrailOutputValidationService
from app.logging_setup import log_agent_step
from app.prompts.prompt_factory import (
    REACT_SYSTEM_PROMPT,
    SINGLE_TOOL_SYSTEM_PROMPT,
    build_styled_user_prompt,
)

MAX_STEP_RESULT_CHARS = 500
MAX_STEP_LOG_CHARS = 300
MAX_STEP_ARGS_CHARS = 200

logger = logging.getLogger(__name__)


class AgentOrchestrationService:
    """
    Orchestrates the agent's reasoning and tool execution.

    Attributes:
        max_react_steps: max tool calls the ReAct agent can make before stopping.
        memory:          MemorySaver gives short-term memory within a conversation thread.
        checker:         GuardrailOutputValidationService validates tool decisions at every step boundary.
    """

    def __init__(self, llm_fast: ChatGroq, llm_react: ChatGroq, tools: list, max_steps: int = 5) -> None:
        self.llm_fast = llm_fast
        self.llm_react = llm_react
        self.tools = tools
        self.max_react_steps = max_steps
        self.memory = MemorySaver()
        self.checker = GuardrailOutputValidationService()

    # ── Week 1: single tool call ──────────────────────────────────────────────

    def execute_single_tool(self, user_request: str, prompt_style: str) -> dict:
        """
        Week 1 mode: one LLM call → picks one tool → executes it → returns result.

        Uses LangChain's bind_tools() for native function calling.
        """
        logger.info("execute_single_tool started — prompt_style=%s", prompt_style)

        llm_with_tools = self.llm_fast.bind_tools(self.tools)
        user_prompt = build_styled_user_prompt(user_request, prompt_style)

        response = llm_with_tools.invoke([
            SystemMessage(content=SINGLE_TOOL_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        if not response.tool_calls:
            return {"mode": "once", "message": response.content}

        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Guardrail: validate tool name + arguments before execution
        ok, msg = self.checker.validate_tool_decision(tool_name, tool_args)
        if not ok:
            logger.warning("Guardrail blocked: %s", msg)
            return {"mode": "once", "error": f"Guardrail blocked: {msg}"}

        tool_map = {t.name: t for t in self.tools}
        tool = tool_map.get(tool_name)
        if not tool:
            return {"mode": "once", "error": f"Unknown tool: {tool_name}"}

        result = tool.invoke(tool_args)
        logger.info("execute_single_tool tool=%s result_len=%d", tool_name, len(str(result)))

        return {
            "mode": "once",
            "prompt_style": prompt_style,
            "selected_tool": tool_name,
            "arguments": tool_args,
            "tool_result": result,
        }

    # ── Week 2/4: ReAct loop via LangGraph ────────────────────────────────────

    def execute_multi_step_agent(self, user_request: str) -> dict:
        """
        Week 2/4 mode: multi-step ReAct loop using LangGraph.

        Flow (handled by LangGraph automatically):
          1. REASON — LLM thinks and picks a tool (AIMessage with tool_calls)
          2. ACT    — LangGraph executes the tool (ToolMessage with result)
          3. OBSERVE — result added to conversation history (MemorySaver)
          4. REPEAT  — LLM sees result and decides next step
          5. DONE   — LLM responds with text (no tool_calls) = finished

        Added on top of LangGraph:
          - trace_id for observability (every step logged to JSONL)
          - Guardrail validation of every tool call in message history
          - recursion_limit prevents infinite loops
        """
        trace_id = uuid.uuid4().hex[:6]
        logger.info("[%s] execute_multi_step_agent started", trace_id)
        log_agent_step("ReactAgent", "start", user_request, "", "started", trace_id=trace_id)

        agent = create_react_agent(
            self.llm_react,
            self.tools,
            prompt=REACT_SYSTEM_PROMPT,
            checkpointer=self.memory,
        )

        config = {
            "configurable": {"thread_id": trace_id},
            "recursion_limit": self.max_react_steps * 2 + 2,
        }

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_request)]},
                config=config,
            )
        except Exception as exc:
            logger.error("[%s] ReAct agent failed: %s", trace_id, exc)
            log_agent_step("ReactAgent", "error", user_request[:200], str(exc)[:300], "failed", trace_id=trace_id)
            return {"mode": "react", "trace_id": trace_id, "error": str(exc)}

        messages = result["messages"]
        steps = self._parse_message_history(messages, trace_id)
        final_message = messages[-1].content if messages else "No response"

        log_agent_step("ReactAgent", "done", "", final_message[:300], "completed", trace_id=trace_id)
        logger.info("[%s] execute_multi_step_agent finished — %d steps", trace_id, len(steps))

        return {
            "mode": "react",
            "trace_id": trace_id,
            "steps": steps,
            "final_message": final_message,
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _parse_message_history(self, messages: list, trace_id: str = "") -> list[dict]:
        """
        Parse LangGraph message history into a readable step log.

        Also runs guardrail validation on every tool call and logs each step
        to agent_steps.jsonl for full observability.
        """
        steps = []
        step_num = 0
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    step_num += 1
                    tool_name = tc["name"]
                    tool_args = tc["args"]

                    # Guardrail check at the output→action boundary
                    ok, guard_msg = self.checker.validate_tool_decision(tool_name, tool_args)
                    log_agent_step("ReactAgent", f"step_{step_num}_guardrail", tool_name, str(ok), guard_msg, trace_id=trace_id)

                    steps.append({
                        "action": "tool_call",
                        "tool": tool_name,
                        "args": tool_args,
                        "thought": msg.content or "",
                        "guardrail_passed": ok,
                    })
                    log_agent_step("ReactAgent", f"step_{step_num}_tool_{tool_name}", str(tool_args)[:MAX_STEP_ARGS_CHARS], "", "called", trace_id=trace_id)

            elif isinstance(msg, ToolMessage):
                steps.append({
                    "action": "observation",
                    "tool": msg.name,
                    "result": str(msg.content)[:MAX_STEP_RESULT_CHARS],
                })
                log_agent_step("ReactAgent", f"step_{step_num}_observe", msg.name, str(msg.content)[:MAX_STEP_LOG_CHARS], "observed", trace_id=trace_id)

            elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                steps.append({
                    "action": "final_response",
                    "content": msg.content,
                })
        return steps

    @staticmethod
    def format_result_as_json(result: dict) -> str:
        """Pretty-print a result dict as JSON."""
        return json.dumps(result, indent=2, default=str)
