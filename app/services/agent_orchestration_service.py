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

    def __init__(self, llm_fast: ChatGroq, llm_react: ChatGroq, tools: list, max_steps: int = 5) -> None:
        self.llm_fast = llm_fast
        self.llm_react = llm_react
        self.tools = tools
        self.max_react_steps = max_steps
        self.memory = MemorySaver()
        self.checker = GuardrailOutputValidationService()


    def execute_single_tool(self, user_request: str, prompt_style: str) -> dict:

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

        logger.info(
            "step=once_llm decision=tool_selected tool=%s prompt_style=%s",
            tool_name, prompt_style,
        )


        ok, msg = self.checker.validate_tool_decision(tool_name, tool_args)
        logger.info(
            "step=guardrail tool=%s decision=%s",
            tool_name, "passed" if ok else f"blocked — {msg}",
        )
        if not ok:
            logger.warning("step=guardrail decision=blocked tool=%s reason=%s", tool_name, msg)
            return {"mode": "once", "error": f"Guardrail blocked: {msg}"}

        tool_map = {t.name: t for t in self.tools}
        tool = tool_map.get(tool_name)
        if not tool:
            logger.error("step=tool_lookup decision=unknown_tool tool=%s", tool_name)
            return {"mode": "once", "error": f"Unknown tool: {tool_name}"}

        logger.info("step=tool_exec decision=executing tool=%s", tool_name)
        result = tool.invoke(tool_args)
        logger.info("step=tool_exec decision=done tool=%s result_len=%d", tool_name, len(str(result)))

        return {
            "mode": "once",
            "prompt_style": prompt_style,
            "selected_tool": tool_name,
            "arguments": tool_args,
            "tool_result": result,
        }


    def execute_multi_step_agent(self, user_request: str) -> dict:
        trace_id = uuid.uuid4().hex[:6]
        logger.info(
            "[%s] step=react_start decision=running input_len=%d max_steps=%d",
            trace_id, len(user_request), self.max_react_steps,
        )
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
            logger.error(
                "[%s] step=react_invoke decision=error err=%s",
                trace_id, exc, exc_info=True,
            )
            log_agent_step("ReactAgent", "error", user_request[:200], str(exc)[:300], "failed", trace_id=trace_id)
            return {"mode": "react", "trace_id": trace_id, "error": str(exc)}

        messages = result["messages"]
        steps = self._parse_message_history(messages, trace_id)
        final_message = messages[-1].content if messages else "No response"

        log_agent_step("ReactAgent", "done", "", final_message[:300], "completed", trace_id=trace_id)
        logger.info(
            "[%s] step=react_done decision=completed steps=%d",
            trace_id, len(steps),
        )

        return {
            "mode": "react",
            "trace_id": trace_id,
            "steps": steps,
            "final_message": final_message,
        }

    def _parse_message_history(self, messages: list, trace_id: str = "") -> list[dict]:
        steps = []
        step_num = 0
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    step_num += 1
                    tool_name = tc["name"]
                    tool_args = tc["args"]


                    ok, guard_msg = self.checker.validate_tool_decision(tool_name, tool_args)
                    log_agent_step("ReactAgent", f"step_{step_num}_guardrail", tool_name, str(ok), guard_msg, trace_id=trace_id)
                    logger.info(
                        "[%s] step=tool_call step_num=%d tool=%s decision=guardrail_%s",
                        trace_id, step_num, tool_name, "passed" if ok else f"blocked — {guard_msg}",
                    )

                    steps.append({
                        "action": "tool_call",
                        "tool": tool_name,
                        "args": tool_args,
                        "thought": msg.content or "",
                        "guardrail_passed": ok,
                    })
                    log_agent_step("ReactAgent", f"step_{step_num}_tool_{tool_name}", str(tool_args)[:MAX_STEP_ARGS_CHARS], "", "called", trace_id=trace_id)

            elif isinstance(msg, ToolMessage):
                result_preview = str(msg.content)[:MAX_STEP_RESULT_CHARS]
                steps.append({
                    "action": "observation",
                    "tool": msg.name,
                    "result": result_preview,
                })
                logger.info(
                    "[%s] step=observe step_num=%d tool=%s decision=observed result_len=%d",
                    trace_id, step_num, msg.name, len(str(msg.content)),
                )
                log_agent_step("ReactAgent", f"step_{step_num}_observe", msg.name, str(msg.content)[:MAX_STEP_LOG_CHARS], "observed", trace_id=trace_id)

            elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                steps.append({
                    "action": "final_response",
                    "content": msg.content,
                })
        return steps

    @staticmethod
    def format_result_as_json(result: dict) -> str:
        return json.dumps(result, indent=2, default=str)
