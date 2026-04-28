
import argparse
import json
import logging

from app.automation_agent import AutomationAgent
from app.logging_setup import configure_logging


class AgentCli:
    """Simple OOP CLI: parse args, run agent, print JSON."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Automation agent (single step or ReAct)"
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
            help="once = one LLM + one tool; react = multi-step loop",
        )
        parser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Console log level",
        )
        parser.add_argument(
            "--workflow",
            choices=["report", "analysis", "tasks"],
            default=None,
            help="Week 3: run a workflow instead of the ReAct agent",
        )
        return parser.parse_args()

    def run(self) -> None:
        args = self.parse_args()
        configure_logging(args.log_level)
        self.logger.info("Start mode=%s prompt_style=%s", args.mode, args.prompt_style)

        try:
            agent = AutomationAgent.create()

            # Week 3: if --workflow is given, run a workflow instead of the agent loop
            if args.workflow:
                self.logger.info("Running workflow: %s", args.workflow)
                result = agent.run_workflow(args.workflow)
            else:
                result = agent.run(
                    user_request=args.request,
                    mode=args.mode,
                    prompt_style=args.prompt_style,
                )

            self.logger.debug("Raw result dict: %s", result)
            print("\n--- RESULT (JSON) ---")
            print(agent.result_as_json(result))
        except Exception as exc:
            self.logger.error("Run failed: %s", exc)
            self.logger.debug("Exception details", exc_info=True)
            print(json.dumps({"success": False, "message": str(exc)}, indent=2))


if __name__ == "__main__":
    AgentCli().run()
