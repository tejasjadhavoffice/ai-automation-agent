"""
Simple OOP entry for the two-week agent.

One class owns: config → Groq client → tools → orchestrator (brain).
`main.py` only parses CLI and calls `AutomationAgent.create().run(...)`.
"""
from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

from app.clients.groq_client import GroqChatClient
from app.config.settings import AppSettings, get_settings
from app.services.agent_orchestration_service import AgentOrchestrationService
from app.services.tool_execution_service import ToolExecutionService

# Project root `.env` (folder above `app/`)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


class AutomationAgent:
    """
    Facade you interact with from `main.py`.

    Attributes are public on purpose so you can read `agent.settings` etc. when learning.
    """

    def __init__(self, settings: AppSettings) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.settings = settings
        self.groq = GroqChatClient(settings=settings)
        self.tools = ToolExecutionService(settings=settings)
        self.orchestrator = AgentOrchestrationService(
            groq_client=self.groq,
            tool_execution_service=self.tools,
        )

    @classmethod
    def create(cls) -> AutomationAgent:
        """Load `.env`, read settings, build the agent — call this once at program start."""
        load_dotenv(_ENV_PATH)
        settings = get_settings()
        agent = cls(settings=settings)
        agent.logger.debug("AutomationAgent created")
        return agent

    def run(self, user_request: str, mode: str, prompt_style: str) -> dict:
        """
        Run Week 1 (`once`) or Week 2 (`react`) flow. Returns the same dict as before.
        """
        self.logger.info("Run requested with mode=%s prompt_style=%s", mode, prompt_style)
        if mode == "once":
            return self.orchestrator.run_once(user_request, prompt_style)
        return self.orchestrator.run_react(user_request, prompt_style)

    def run_workflow(self, workflow_name: str) -> dict:
        """
        Week 3: run one of the three automation workflows by name.
        Imports are inside the method so the rest of the agent still works
        even if workflow files are not present.
        """
        from app.workflows.analysis_workflow import AnalysisWorkflow
        from app.workflows.report_workflow import ReportWorkflow
        from app.workflows.task_scheduler_workflow import TaskSchedulerWorkflow

        workflows = {
            "report": ReportWorkflow,
            "analysis": AnalysisWorkflow,
            "tasks": TaskSchedulerWorkflow,
        }
        cls = workflows.get(workflow_name)
        if not cls:
            raise ValueError(
                f"Unknown workflow: '{workflow_name}'. Choose from: {list(workflows.keys())}"
            )
        workflow = cls(groq_api_key=self.settings.groq_api_key)
        return workflow.run()

    def result_as_json(self, result: dict) -> str:
        """Pretty-print a result dict (same string as `AgentOrchestrationService.format_output`)."""
        return self.orchestrator.format_output(result)
