import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from app.clients.groq_client import GroqChatClient
from app.config.settings import get_settings
from app.services.agent_orchestration_service import AgentOrchestrationService
from app.services.tool_execution_service import ToolExecutionService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 1 AI automation agent")
    parser.add_argument("--request", required=True, help="Plain English request")
    parser.add_argument(
        "--prompt-style",
        default="zero-shot",
        choices=["zero-shot", "few-shot", "cot"],
        help="Prompt style for experiment",
    )
    return parser.parse_args()


def main() -> None:
    backend_dir = Path(__file__).resolve().parents[1]
    load_dotenv(backend_dir / ".env")
    args = parse_args()

    settings = get_settings()
    orchestrator = AgentOrchestrationService(
        groq_client=GroqChatClient(settings=settings),
        tool_execution_service=ToolExecutionService(settings=settings),
    )

    try:
        print(f"Running agent with prompt style: {args.prompt_style}")
        result = orchestrator.run_once(
            user_request=args.request,
            prompt_style=args.prompt_style,
        )
        print(orchestrator.format_output(result))
    except Exception as exc:
        print(f"Agent failed: {exc}")
        error_payload = {
            "success": False,
            "message": str(exc),
        }
        print(json.dumps(error_payload, indent=2))


if __name__ == "__main__":
    main()
