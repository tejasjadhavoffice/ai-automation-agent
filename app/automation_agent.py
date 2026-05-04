"""
automation_agent.py — Facade pattern.

One class wires everything together:
  config → LLM client → tools → orchestrator

main.py only calls AutomationAgent.initialize().execute() — it doesn't
need to know about LangChain, LangGraph, or individual tools.
"""
from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

from app.clients.groq_client import LLMClient
from app.config.settings import Settings, get_settings
from app.services.agent_orchestration_service import AgentOrchestrationService
from app.services.tool_execution_service import ToolRegistryService

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

logger = logging.getLogger(__name__)


class AutomationAgent:
    """
    Facade: the single entry point for all agent operations.

    Why Facade pattern?
      - main.py calls agent.run() — simple, clean
      - All internal wiring (LLM, tools, orchestrator) is hidden
      - Easy to swap any component without changing the CLI
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.llm_client = LLMClient(settings=settings)
        self.tool_registry = ToolRegistryService()
        self.orchestrator = AgentOrchestrationService(
            llm_fast=self.llm_client.llm,
            llm_react=self.llm_client.llm,
            tools=self.tool_registry.get_registered_tools(),
            max_steps=settings.agent_max_steps,
        )

    @classmethod
    def initialize(cls) -> AutomationAgent:
        """Load .env, read settings, build the agent — call once at startup."""
        load_dotenv(_ENV_PATH)
        settings = get_settings()
        agent = cls(settings=settings)
        logger.debug("AutomationAgent initialized")
        return agent

    def execute(self, user_request: str, mode: str, prompt_style: str = "zero-shot") -> dict:
        """
        Execute the agent in the requested mode.

        Modes:
          once  → Week 1: single LLM call + one tool
          react → Week 2/4: LangGraph ReAct loop with memory
        """
        logger.info("Execute mode=%s prompt_style=%s", mode, prompt_style)

        if mode == "once":
            return self.orchestrator.execute_single_tool(user_request, prompt_style)
        return self.orchestrator.execute_multi_step_agent(user_request)

    def execute_workflow(self, workflow_name: str) -> dict:
        """
        Week 3: run one of the three automation workflows by name.

        Lazy imports so the rest of the agent works even if workflow
        files have issues.
        """
        from app.workflows.analysis_workflow import AnalysisWorkflow
        from app.workflows.report_workflow import ReportWorkflow
        from app.workflows.task_router_workflow import TaskRouterWorkflow

        workflows = {
            "report": ReportWorkflow,
            "analysis": AnalysisWorkflow,
            "tasks": TaskRouterWorkflow,
        }
        cls = workflows.get(workflow_name)
        if not cls:
            raise ValueError(
                f"Unknown workflow: '{workflow_name}'. Choose from: {list(workflows.keys())}"
            )
        workflow = cls(groq_api_key=self.settings.groq_api_key)
        return workflow.run()

    def format_result_as_json(self, result: dict) -> str:
        """Pretty-print a result dict."""
        return self.orchestrator.format_result_as_json(result)
