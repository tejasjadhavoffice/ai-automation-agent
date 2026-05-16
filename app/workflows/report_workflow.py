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

    def __init__(self, groq_api_key: str) -> None:
        super().__init__(groq_api_key)
        self.checker = GuardrailOutputValidationService()
        self.client = Groq(api_key=groq_api_key)

    def run(self) -> dict:
        today = date.today().isoformat()
        output_file = f"report_{today}.txt"

        if self._is_already_processed(output_file):
            logger.info(
                "workflow=ReportWorkflow step=idempotency input=%s decision=skip output=already_exists",
                output_file,
            )
            log_agent_step("ReportWorkflow", "idempotency_check", output_file, "skipped", "already done")
            return {"success": True, "message": "Already done — report exists", "file": output_file}

        log_agent_step("ReportWorkflow", "read_data", DATA_FILE, "", "reading file")
        try:
            raw = Path(DATA_FILE).read_text(encoding="utf-8")
            logger.info(
                "workflow=ReportWorkflow step=read_data input=%s decision=ok output=chars=%d",
                DATA_FILE, len(raw),
            )
        except FileNotFoundError:
            log_agent_step("ReportWorkflow", "read_data", DATA_FILE, "error", "file not found")
            logger.error(
                "workflow=ReportWorkflow step=read_data input=%s decision=file_not_found",
                DATA_FILE,
            )
            return {"success": False, "message": f"Failure: data file not found: {DATA_FILE}"}

        logger.info(
            "workflow=ReportWorkflow step=llm_generate decision=calling_llm input_chars=%d",
            len(raw),
        )
        log_agent_step("ReportWorkflow", "generate_report", raw[:200], "", "calling LLM")
        summary = self._generate_report_from_llm(raw)


        ok, msg = self.checker.validate_report_output(summary)
        log_agent_step("ReportWorkflow", "guardrail", summary[:200], str(ok), msg)
        if not ok:
            logger.warning(
                "workflow=ReportWorkflow step=guardrail decision=blocked input_len=%d reason=%s",
                len(summary), msg,
            )
            return {"success": False, "message": f"Guardrail failed: {msg}"}
        logger.info(
            "workflow=ReportWorkflow step=guardrail decision=passed summary_len=%d",
            len(summary),
        )

        self._save_result_to_file(output_file, summary)
        log_agent_step("ReportWorkflow", "save_report", output_file, "saved", "done")
        logger.info(
            "workflow=ReportWorkflow step=save decision=saved output=%s",
            output_file,
        )
        return {
            "success": True,
            "message": "Report generated",
            "file": output_file,
            "preview": summary[:300],
        }

    def _generate_report_from_llm(self, raw_data: str) -> str:
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
            logger.error(
                "workflow=ReportWorkflow step=llm_generate decision=error err=%s",
                exc, exc_info=True,
            )
            return ""
