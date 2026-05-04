"""
analysis_workflow.py — Workflow 2

Steps:
  1. Idempotency check — if today's analysis already exists, skip.
  2. Read raw sales data from data/sales_data.txt.
  3. Failure simulation — if data is too short, stop gracefully.
  4. Call Groq LLM to find trends and key numbers in the data.
  5. Guardrail — validate the LLM output.
  6. Save analysis to data/reports/analysis_<date>.txt.
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


class AnalysisWorkflow(BaseWorkflow):
    """Workflow 2: Raw data → LLM finds trends → save analysis."""

    def __init__(self, groq_api_key: str) -> None:
        super().__init__(groq_api_key)
        self.checker = GuardrailOutputValidationService()
        self.client = Groq(api_key=groq_api_key)

    def run(self) -> dict:
        today = date.today().isoformat()
        output_file = f"analysis_{today}.txt"

        # Step 1: Idempotency — skip if analysis already done today
        if self._is_already_processed(output_file):
            logger.info("Analysis already exists for today — skipping")
            log_agent_step("AnalysisWorkflow", "idempotency_check", output_file, "skipped", "already done")
            return {"success": True, "message": "Already done — analysis exists", "file": output_file}

        # Step 2: Read raw data
        log_agent_step("AnalysisWorkflow", "read_data", DATA_FILE, "", "reading")
        try:
            raw = Path(DATA_FILE).read_text(encoding="utf-8")
        except FileNotFoundError:
            log_agent_step("AnalysisWorkflow", "read_data", DATA_FILE, "error", "file not found")
            return {"success": False, "message": f"Failure: data file not found: {DATA_FILE}"}

        # Step 3: Failure simulation — empty or trivial data is rejected
        if len(raw.strip()) < 10:
            log_agent_step("AnalysisWorkflow", "validate_data", raw, "error", "data too short")
            logger.error("Data too short to analyse — simulated failure")
            return {"success": False, "message": "Failure: data is too short for analysis"}

        # Step 4: Ask LLM to analyse trends
        log_agent_step("AnalysisWorkflow", "analyse", raw[:200], "", "calling LLM")
        analysis = self._analyse_data_with_llm(raw)

        # Step 5: Guardrail — validate output
        ok, msg = self.checker.validate_analysis_output(analysis)
        log_agent_step("AnalysisWorkflow", "guardrail", analysis[:200], str(ok), msg)
        if not ok:
            logger.warning("Guardrail blocked output: %s", msg)
            return {"success": False, "message": f"Guardrail failed: {msg}"}

        # Step 6: Save
        self._save_result_to_file(output_file, analysis)
        log_agent_step("AnalysisWorkflow", "save_analysis", output_file, "saved", "done")
        logger.info("Analysis saved to %s", output_file)
        return {
            "success": True,
            "message": "Analysis complete",
            "file": output_file,
            "preview": analysis[:300],
        }

    def _analyse_data_with_llm(self, raw_data: str) -> str:
        """Ask Groq LLM to identify key trends and numbers in the data."""
        try:
            response = self.client.chat.completions.create(
                model=get_settings().groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a data analyst. Analyse the sales data and write "
                            "4-6 bullet points covering: top performers, trends, and key numbers."
                        ),
                    },
                    {"role": "user", "content": raw_data[:MAX_LLM_INPUT_CHARS]},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return ""
