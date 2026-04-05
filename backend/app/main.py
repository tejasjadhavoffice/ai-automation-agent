import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from app.clients.groq_client import GroqChatClient
from app.config.settings import get_settings
from app.services.agent_orchestration_service import AgentOrchestrationService
from app.services.tool_execution_service import ToolExecutionService


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    p = argparse.ArgumentParser(description="Week 1–2 automation agent")
    p.add_argument("--request", required=True)
    p.add_argument(
        "--prompt-style",
        default="zero-shot",
        choices=["zero-shot", "few-shot", "cot"],
    )
    p.add_argument(
        "--mode",
        default="react",
        choices=["once", "react"],
        help="once=single tool call (Week 1); react=ReAct loop (Week 2)",
    )
    args = p.parse_args()

    settings = get_settings()
    orch = AgentOrchestrationService(
        GroqChatClient(settings=settings),
        ToolExecutionService(settings=settings),
    )

    print(f"[start] mode={args.mode} prompt_style={args.prompt_style}")
    try:
        if args.mode == "once":
            out = orch.run_once(args.request, args.prompt_style)
        else:
            out = orch.run_react(args.request, args.prompt_style)
        print("\n--- RESULT (JSON) ---")
        print(orch.format_output(out))
    except Exception as exc:
        print(f"[error] {exc}")
        print(json.dumps({"success": False, "message": str(exc)}, indent=2))


if __name__ == "__main__":
    main()
