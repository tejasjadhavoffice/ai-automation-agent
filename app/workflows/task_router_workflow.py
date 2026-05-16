import json
import logging
import re
from pathlib import Path

from groq import Groq

from app.config.settings import get_settings
from app.guardrails.guardrail_checker import GuardrailOutputValidationService
from app.logging_setup import log_agent_step
from app.workflows.base_workflow import BaseWorkflow

TASKS_FILE = "data/tasks.txt"
PROCESSED_LOG = Path("data/processed_tasks.txt")

logger = logging.getLogger(__name__)


class TaskRouterWorkflow(BaseWorkflow):
    def __init__(self, groq_api_key: str) -> None:
        super().__init__(groq_api_key)
        self.checker = GuardrailOutputValidationService()
        self.client = Groq(api_key=groq_api_key)

    def run(self) -> dict:
        log_agent_step("TaskRouterWorkflow", "read_tasks", TASKS_FILE, "", "reading")
        try:
            raw = Path(TASKS_FILE).read_text(encoding="utf-8")
        except FileNotFoundError:
            log_agent_step("TaskRouterWorkflow", "read_tasks", TASKS_FILE, "error", "not found")
            logger.error(
                "workflow=TaskRouterWorkflow step=read_tasks input=%s decision=file_not_found",
                TASKS_FILE,
            )
            return {"success": False, "message": f"Failure: task file not found: {TASKS_FILE}"}

        tasks = [line.strip() for line in raw.splitlines() if line.strip()]
        logger.info(
            "workflow=TaskRouterWorkflow step=read_tasks input=%s decision=ok output=tasks=%d",
            TASKS_FILE, len(tasks),
        )

        done_tasks = self._load_processed_tasks()
        results = []

        for task in tasks:
            if task in done_tasks:
                logger.info(
                    "workflow=TaskRouterWorkflow step=task_check input=%r decision=skip output=already_done",
                    task,
                )
                log_agent_step("TaskRouterWorkflow", "skip_task", task, "skipped", "already done")
                results.append({"task": task, "status": "skipped — already done"})
                continue

            log_agent_step("TaskRouterWorkflow", "route_task", task, "", "calling LLM")
            tool_name = self._route_task_to_tool(task)

            ok, msg = self.checker.validate_task_assignment(tool_name)
            log_agent_step("TaskRouterWorkflow", "guardrail", task, str(ok), f"tool={tool_name} | {msg}")

            if not ok:
                logger.warning(
                    "workflow=TaskRouterWorkflow step=guardrail input=%r decision=blocked tool=%s reason=%s",
                    task, tool_name, msg,
                )
                results.append({"task": task, "tool": tool_name, "status": f"guardrail_failed: {msg}"})
                continue

            logger.info(
                "workflow=TaskRouterWorkflow step=route input=%r decision=routed output=tool=%s",
                task, tool_name,
            )
            results.append({"task": task, "tool": tool_name, "status": "routed"})
            self._mark_task_as_done(task)

        log_agent_step(
            "TaskRouterWorkflow", "complete",
            str(len(tasks)), str(len(results)), f"{len(results)} tasks processed"
        )
        logger.info(
            "workflow=TaskRouterWorkflow step=complete decision=done input=tasks=%d output=results=%d",
            len(tasks), len(results),
        )
        return {"success": True, "message": "Task routing complete", "results": results}

    def _route_task_to_tool(self, task: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=get_settings().groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a task router. Given a task description, "
                            "return ONLY JSON with one key: tool_name.\n"
                            "Allowed values: read_file, fetch_data, summarise_text, send_email, no_tool.\n"
                            'Example: {"tool_name": "read_file"}'
                        ),
                    },
                    {"role": "user", "content": f"Task: {task}"},
                ],
                temperature=0.1,
            )
            content = response.choices[0].message.content or "{}"
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return data.get("tool_name", "no_tool")
            return "no_tool"
        except Exception as exc:
            logger.error(
                "workflow=TaskRouterWorkflow step=llm_route input=%r decision=error err=%s",
                task, exc, exc_info=True,
            )
            return "no_tool"

    def _load_processed_tasks(self) -> set:
        if not PROCESSED_LOG.exists():
            return set()
        return set(PROCESSED_LOG.read_text(encoding="utf-8").splitlines())

    def _mark_task_as_done(self, task: str) -> None:
        with open(PROCESSED_LOG, "a", encoding="utf-8") as f:
            f.write(task + "\n")
