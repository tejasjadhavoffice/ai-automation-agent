
import argparse
import json
import logging

from app.automation_agent import AutomationAgent
from app.logging_setup import setup_console_logging

logger = logging.getLogger(__name__)

class AgentRunner:

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
        parser.add_argument(
            "--serve",
            action="store_true",
            help="Start the FastAPI server instead of running a CLI command",
        )
        return parser.parse_args()

    def start(self) -> None:
        args = self.parse_arguments()
        setup_console_logging()

        if args.serve:
            logger.info("step=dispatch decision=starting_api_server")
            import uvicorn
            from app.api import app  
            uvicorn.run(app, host="0.0.0.0", port=8000)
            return

        if args.workflow:
            logger.info(
                "step=dispatch decision=workflow workflow=%s", args.workflow
            )
        else:
            logger.info(
                "step=dispatch decision=agent mode=%s prompt_style=%s input_len=%d",
                args.mode, args.prompt_style, len(args.request),
            )

        try:
            agent = AutomationAgent.initialize()

            if args.workflow:
                result = agent.execute_workflow(args.workflow)
            else:
                result = agent.execute(
                    user_request=args.request,
                    mode=args.mode,
                    prompt_style=args.prompt_style,
                )

            result_json = agent.format_result_as_json(result)
            logger.info(
                "step=run_done decision=success result_keys=%s\n%s",
                list(result.keys()), result_json,
            )
        except Exception as exc:
            failure_json = json.dumps({"success": False, "message": str(exc)}, indent=2)
            logger.error(
                "step=run_done decision=error err=%s\n%s",
                exc, failure_json, exc_info=True,
            )


if __name__ == "__main__":
    AgentRunner().start()
