
import argparse
import json
import logging

from app.automation_agent import AutomationAgent
from app.logging_setup import setup_console_logging

logger = logging.getLogger(__name__)


class AgentRunner:
    """Entry point: parse arguments, run agent, print JSON."""

    def __init__(self) -> None:
        pass

    def parse_arguments(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="AI Automation Agent — LangChain + LangGraph"
        )
        parser.add_argument("--request", default="", help="What you want the agent to do")
        parser.add_argument(
            "--prompt-style",
            default="zero-shot",
            choices=["zero-shot", "few-shot", "cot"],
        )
        parser.add_argument(
            "--mode",
            default="react",
            choices=["once", "react"],
            help="once = Week 1 single tool; react = Week 2/4 LangGraph ReAct loop",
        )
        parser.add_argument(
            "--workflow",
            choices=["report", "analysis", "tasks"],
            default=None,
            help="Week 3: run a workflow instead of the ReAct agent",
        )
        return parser.parse_args()

    def start(self) -> None:
        args = self.parse_arguments()
        setup_console_logging()
        logger.info("Start mode=%s prompt_style=%s", args.mode, args.prompt_style)

        try:
            agent = AutomationAgent.initialize()

            if args.workflow:
                logger.info("Running workflow: %s", args.workflow)
                result = agent.execute_workflow(args.workflow)
            else:
                result = agent.execute(
                    user_request=args.request,
                    mode=args.mode,
                    prompt_style=args.prompt_style,
                )

            print("\n--- RESULT (JSON) ---")
            print(agent.format_result_as_json(result))
        except Exception as exc:
            logger.error("Run failed: %s", exc)
            logger.debug("Exception details", exc_info=True)
            print(json.dumps({"success": False, "message": str(exc)}, indent=2))


if __name__ == "__main__":
    AgentRunner().start()
