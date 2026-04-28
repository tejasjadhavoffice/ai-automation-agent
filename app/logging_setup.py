import json
import logging
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("logs")


def configure_logging(level_name: str = "INFO") -> None:
    """Configure simple console logging for the whole app."""
    LOG_DIR.mkdir(exist_ok=True)
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def log_step(workflow: str, step: str, input_data: str, output: str, decision: str) -> None:
    """
    Write one structured JSON line to logs/agent_steps.jsonl.

    Every workflow step calls this so we have full observability:
    what went in, what came out, and what decision was made.

    Args:
        workflow: name of the workflow (e.g. "ReportWorkflow")
        step:     name of the step  (e.g. "guardrail", "save_report")
        input_data: what the step received (truncated to 300 chars)
        output:   what the step produced  (truncated to 300 chars)
        decision: short label for what happened (e.g. "OK", "skipped", "error")
    """
    LOG_DIR.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "workflow": workflow,
        "step": step,
        "input": str(input_data)[:300],
        "output": str(output)[:300],
        "decision": decision,
    }
    log_file = LOG_DIR / "agent_steps.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
