import json
import logging
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("logs")
MAX_LOG_CHARS = 300

logger = logging.getLogger(__name__)


def setup_console_logging() -> None:
    """Configure console logging for the whole app — always DEBUG level."""
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(name)s - %(module)s:%(lineno)d - %(funcName)s - %(message)s",
    )


def log_agent_step(
    workflow: str,
    step: str,
    input_data: str,
    output: str,
    decision: str,
    trace_id: str = "",
) -> None:
    """
    Write one structured JSON line to logs/agent_steps.jsonl.

    Args:
        workflow:   name of the workflow / agent (e.g. "ReactAgent")
        step:       name of the step (e.g. "guardrail", "tool_exec")
        input_data: what the step received (truncated to 300 chars)
        output:     what the step produced  (truncated to 300 chars)
        decision:   short label for what happened ("OK", "skipped", "error")
        trace_id:   short run ID so you can follow one run in the logs
    """
    LOG_DIR.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "workflow": workflow,
        "step": step,
        "input": str(input_data)[:MAX_LOG_CHARS],
        "output": str(output)[:MAX_LOG_CHARS],
        "decision": decision,
    }
    log_file = LOG_DIR / "agent_steps.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
