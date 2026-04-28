"""
task_scheduler_workflow.py — Workflow 3

Steps:
  1. Read task list from data/tasks.txt (one task per line).
  2. Load already-processed tasks from data/processed_tasks.txt (idempotency).
  3. For each new task:
       a. Ask LLM which tool should handle it.
       b. Guardrail — validate the tool name before scheduling.
       c. Log the assignment and mark the task as done.

Failure simulation:
  If the LLM returns an unknown tool name, the guardrail blocks it
  and logs the failure — no crash, just a safe skip.
"""

import json
import logging
import re
from pathlib import Path

from groq import Groq

from app.guardrails.guardrail_checker import GuardrailChecker
from app.logging_setup import log_step
from app.workflows.base_workflow import BaseWorkflow

TASKS_FILE = "data/tasks.txt"
PROCESSED_LOG = Path("data/processed_tasks.txt")


class TaskSchedulerWorkflow(BaseWorkflow):
    """Workflow 3: Read task list → LLM assigns each task to a tool → log done tasks."""

    def __init__(self, groq_api_key: str) -> None:
        super().__init__(groq_api_key)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.checker = GuardrailChecker()
        self.client = Groq(api_key=groq_api_key)

    def run(self) -> dict:
        # Step 1: Read task list
        log_step("TaskSchedulerWorkflow", "read_tasks", TASKS_FILE, "", "reading")
        try:
            raw = Path(TASKS_FILE).read_text(encoding="utf-8")
        except FileNotFoundError:
            log_step("TaskSchedulerWorkflow", "read_tasks", TASKS_FILE, "error", "not found")
            return {"success": False, "message": f"Failure: task file not found: {TASKS_FILE}"}

        tasks = [line.strip() for line in raw.splitlines() if line.strip()]

        # Step 2: Load already-processed tasks (idempotency)
        done_tasks = self._load_processed()
        results = []

        # Step 3: Process each task
        for task in tasks:
            if task in done_tasks:
                self.logger.info("Skipping already-done task: %s", task)
                log_step("TaskSchedulerWorkflow", "skip_task", task, "skipped", "already done")
                results.append({"task": task, "status": "skipped — already done"})
                continue

            # Ask LLM which tool to use
            log_step("TaskSchedulerWorkflow", "assign_tool", task, "", "calling LLM")
            tool_name = self._assign_tool(task)

            # Guardrail — validate the tool assignment
            ok, msg = self.checker.check_task_assignment(tool_name)
            log_step("TaskSchedulerWorkflow", "guardrail", task, str(ok), f"tool={tool_name} | {msg}")

            if not ok:
                # FAILURE SIMULATION: LLM returned an invalid tool — blocked by guardrail
                self.logger.warning("Guardrail blocked task '%s': %s", task, msg)
                results.append({"task": task, "tool": tool_name, "status": f"guardrail_failed: {msg}"})
                continue

            self.logger.info("Task scheduled: '%s' → tool: %s", task, tool_name)
            results.append({"task": task, "tool": tool_name, "status": "scheduled"})
            self._mark_done(task)

        log_step(
            "TaskSchedulerWorkflow", "complete",
            str(len(tasks)), str(len(results)), f"{len(results)} tasks processed"
        )
        return {"success": True, "message": "Task scheduling complete", "results": results}

    def _assign_tool(self, task: str) -> str:
        """Call Groq LLM to decide which tool handles this task. Returns tool_name string."""
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
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
            self.logger.error("LLM call failed for task '%s': %s", task, exc)
            return "no_tool"

    def _load_processed(self) -> set:
        """Read data/processed_tasks.txt and return a set of already-done task strings."""
        if not PROCESSED_LOG.exists():
            return set()
        return set(PROCESSED_LOG.read_text(encoding="utf-8").splitlines())

    def _mark_done(self, task: str) -> None:
        """Append the task to data/processed_tasks.txt so re-runs skip it."""
        with open(PROCESSED_LOG, "a", encoding="utf-8") as f:
            f.write(task + "\n")
