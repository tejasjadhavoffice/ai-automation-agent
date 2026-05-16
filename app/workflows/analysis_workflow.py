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
    def __init__(self, groq_api_key: str) -> None:
        super().__init__(groq_api_key)
        self.checker = GuardrailOutputValidationService()
        self.client = Groq(api_key=groq_api_key)

    def run(self) -> dict:
        today = date.today().isoformat()
        output_file = f"analysis_{today}.txt"

        if self._is_already_processed(output_file):
            logger.info(
                "workflow=AnalysisWorkflow step=idempotency input=%s decision=skip output=already_exists",
                output_file,
            )
            log_agent_step("AnalysisWorkflow", "idempotency_check", output_file, "skipped", "already done")
            return {"success": True, "message": "Already done — analysis exists", "file": output_file}

        log_agent_step("AnalysisWorkflow", "read_data", DATA_FILE, "", "reading")
        try:
            raw = Path(DATA_FILE).read_text(encoding="utf-8")
            logger.info(
                "workflow=AnalysisWorkflow step=read_data input=%s decision=ok output=chars=%d",
                DATA_FILE, len(raw),
            )
        except FileNotFoundError:
            log_agent_step("AnalysisWorkflow", "read_data", DATA_FILE, "error", "file not found")
            logger.error(
                "workflow=AnalysisWorkflow step=read_data input=%s decision=file_not_found",
                DATA_FILE,
            )
            return {"success": False, "message": f"Failure: data file not found: {DATA_FILE}"}

        if len(raw.strip()) < 10:
            log_agent_step("AnalysisWorkflow", "validate_data", raw, "error", "data too short")
            logger.error(
                "workflow=AnalysisWorkflow step=validate_data input_len=%d decision=data_too_short",
                len(raw.strip()),
            )
            return {"success": False, "message": "Failure: data is too short for analysis"}

        logger.info(
            "workflow=AnalysisWorkflow step=llm_analyse decision=calling_llm input_chars=%d",
            len(raw),
        )
        log_agent_step("AnalysisWorkflow", "analyse", raw[:200], "", "calling LLM")
        analysis = self._analyse_data_with_llm(raw)

        ok, msg = self.checker.validate_analysis_output(analysis)
        log_agent_step("AnalysisWorkflow", "guardrail", analysis[:200], str(ok), msg)
        if not ok:
            logger.warning(
                "workflow=AnalysisWorkflow step=guardrail decision=blocked input_len=%d reason=%s",
                len(analysis), msg,
            )
            return {"success": False, "message": f"Guardrail failed: {msg}"}
        logger.info(
            "workflow=AnalysisWorkflow step=guardrail decision=passed analysis_len=%d",
            len(analysis),
        )

        self._save_result_to_file(output_file, analysis)
        log_agent_step("AnalysisWorkflow", "save_analysis", output_file, "saved", "done")
        logger.info(
            "workflow=AnalysisWorkflow step=save decision=saved output=%s",
            output_file,
        )
        return {
            "success": True,
            "message": "Analysis complete",
            "file": output_file,
            "preview": analysis[:300],
        }

    def _analyse_data_with_llm(self, raw_data: str) -> str:
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
            logger.error(
                "workflow=AnalysisWorkflow step=llm_analyse decision=error err=%s",
                exc, exc_info=True,
            )
            return ""
