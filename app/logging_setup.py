import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("logs")
MAX_LOG_CHARS = 300

_LOG_FORMAT = "%(asctime)s | %(levelname)-5s | %(filename)s:%(lineno)d | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_console_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.WARNING)

    root.handlers.clear()
    root.addHandler(console)

    for name in ("app", "__main__"):
        logging.getLogger(name).setLevel(logging.INFO)


def log_agent_step(
    workflow: str,
    step: str,
    input_data: str,
    output: str,
    decision: str,
    trace_id: str = "",
) -> None:
 
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
