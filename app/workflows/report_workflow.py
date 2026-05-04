"""
report_workflow.py — Workflow 1

Steps:
  1. Idempotency check — if today's report already exists, skip.
  2. Read raw sales data from data/sales_data.txt.
  3. Call Groq LLM to write a business report from that data.
  4. Guardrail — validate the LLM output before saving.
  5. Save report to data/reports/report_<date>.txt.

Failure simulation:
  If data/sales_data.txt is missing, the workflow returns success=False gracefully.
"""

import logging
from datetime import date
from pathlib import Path

from groq import Groq

from app.config.settings import get_settings
from app.guardrails.guardrail_checker import GuardrailOutputValidationService
from app.logging_setup import log_agent_step
from app.workflows.base_workflow import BaseWorkflow

DATA_FILE = "data/sales_data.txt"
MAX_LLM_INPUT_CHARS = 3000

logger = logging.getLogger(__name__)


class ReportWorkflow(BaseWorkflow):
    """Workflow 1: Raw data → LLM report → save to file."""

    def __init__(self, groq_api_key: str) -> None:
        super().__init__(groq_api_key)
        self.checker = GuardrailOutputValidationService()
        self.client = Groq(api_key=groq_api_key)

    def run(self) -> dict:
        today = date.today().isoformat()
        output_file = f"report_{today}.txt"

        # Step 1: Idempotency — skip if report already exists for today
        if self._is_already_processed(output_file):
            logger.info("Report already exists for today — skipping")
            log_agent_step("ReportWorkflow", "idempotency_check", output_file, "skipped", "already done")
            return {"success": True, "message": "Already done — report exists", "file": output_file}

        # Step 2: Read raw data
        log_agent_step("ReportWorkflow", "read_data", DATA_FILE, "", "reading file")
        try:
            raw = Path(DATA_FILE).read_text(encoding="utf-8")
        except FileNotFoundError:
            # FAILURE SIMULATION: data file is missing — return gracefully
            log_agent_step("ReportWorkflow", "read_data", DATA_FILE, "error", "file not found")
            logger.error("Data file missing: %s", DATA_FILE)
            return {"success": False, "message": f"Failure: data file not found: {DATA_FILE}"}

        # Step 3: Ask LLM to write a report
        log_agent_step("ReportWorkflow", "generate_report", raw[:200], "", "calling LLM")
        summary = self._generate_report_from_llm(raw)

        # Step 4: Guardrail — validate before saving
        ok, msg = self.checker.validate_report_output(summary)
        log_agent_step("ReportWorkflow", "guardrail", summary[:200], str(ok), msg)
        if not ok:
            logger.warning("Guardrail blocked output: %s", msg)
            return {"success": False, "message": f"Guardrail failed: {msg}"}

        # Step 5: Save
        self._save_result_to_file(output_file, summary)
        log_agent_step("ReportWorkflow", "save_report", output_file, "saved", "done")
        logger.info("Report saved to %s", output_file)
        return {
            "success": True,
            "message": "Report generated",
            "file": output_file,
            "preview": summary[:300],
        }

    def _generate_report_from_llm(self, raw_data: str) -> str:
        """Ask Groq LLM to turn raw sales data into a readable business report."""
        try:
            response = self.client.chat.completions.create(
                model=get_settings().groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a business analyst. Write a clear 5-7 sentence "
                            "report based on the sales data provided. Include key numbers."
                        ),
                    },
                    {"role": "user", "content": raw_data[:MAX_LLM_INPUT_CHARS]},
                ],
                temperature=0.3,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return ""
