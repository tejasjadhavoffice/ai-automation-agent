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

from app.guardrails.guardrail_checker import GuardrailChecker
from app.logging_setup import log_step
from app.workflows.base_workflow import BaseWorkflow

DATA_FILE = "data/sales_data.txt"


class ReportWorkflow(BaseWorkflow):
    """Workflow 1: Raw data → LLM report → save to file."""

    def __init__(self, groq_api_key: str) -> None:
        super().__init__(groq_api_key)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.checker = GuardrailChecker()
        self.client = Groq(api_key=groq_api_key)

    def run(self) -> dict:
        today = date.today().isoformat()
        output_file = f"report_{today}.txt"

        # Step 1: Idempotency — skip if report already exists for today
        if self._output_exists(output_file):
            self.logger.info("Report already exists for today — skipping")
            log_step("ReportWorkflow", "idempotency_check", output_file, "skipped", "already done")
            return {"success": True, "message": "Already done — report exists", "file": output_file}

        # Step 2: Read raw data
        log_step("ReportWorkflow", "read_data", DATA_FILE, "", "reading file")
        try:
            raw = Path(DATA_FILE).read_text(encoding="utf-8")
        except FileNotFoundError:
            # FAILURE SIMULATION: data file is missing — return gracefully
            log_step("ReportWorkflow", "read_data", DATA_FILE, "error", "file not found")
            self.logger.error("Data file missing: %s", DATA_FILE)
            return {"success": False, "message": f"Failure: data file not found: {DATA_FILE}"}

        # Step 3: Ask LLM to write a report
        log_step("ReportWorkflow", "generate_report", raw[:200], "", "calling LLM")
        summary = self._call_llm(raw)

        # Step 4: Guardrail — validate before saving
        ok, msg = self.checker.check_report(summary)
        log_step("ReportWorkflow", "guardrail", summary[:200], str(ok), msg)
        if not ok:
            self.logger.warning("Guardrail blocked output: %s", msg)
            return {"success": False, "message": f"Guardrail failed: {msg}"}

        # Step 5: Save
        self._save_output(output_file, summary)
        log_step("ReportWorkflow", "save_report", output_file, "saved", "done")
        self.logger.info("Report saved to %s", output_file)
        return {
            "success": True,
            "message": "Report generated",
            "file": output_file,
            "preview": summary[:300],
        }

    def _call_llm(self, raw_data: str) -> str:
        """Ask Groq LLM to turn raw sales data into a readable business report."""
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a business analyst. Write a clear 5-7 sentence "
                            "report based on the sales data provided. Include key numbers."
                        ),
                    },
                    {"role": "user", "content": raw_data[:3000]},
                ],
                temperature=0.3,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            self.logger.error("LLM call failed: %s", exc)
            return ""
