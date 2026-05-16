import json
from typing import Literal

from guardrails import Guard
from pydantic import BaseModel, field_validator

MIN_REPORT_LENGTH = 30
MIN_ANALYSIS_LENGTH = 20

AllowedToolName = Literal[
    "read_file", "send_email", "fetch_data", "summarise_text", "no_tool"
]

class ReportOutput(BaseModel):
    summary: str

    @field_validator("summary")
    @classmethod
    def summary_long_enough(cls, v: str) -> str:
        if len(v.strip()) < MIN_REPORT_LENGTH:
            raise ValueError(f"Report summary too short (minimum {MIN_REPORT_LENGTH} characters)")
        return v


class AnalysisOutput(BaseModel):
    analysis: str

    @field_validator("analysis")
    @classmethod
    def analysis_not_empty(cls, v: str) -> str:
        if len(v.strip()) < MIN_ANALYSIS_LENGTH:
            raise ValueError(f"Analysis output too short (minimum {MIN_ANALYSIS_LENGTH} characters)")
        return v


class TaskAssignment(BaseModel):
    tool_name: str

    @field_validator("tool_name")
    @classmethod
    def must_be_valid_tool(cls, v: str) -> str:
        allowed = {"read_file", "fetch_data", "summarise_text", "send_email", "no_tool"}
        if v not in allowed:
            raise ValueError(f"'{v}' is not a valid tool. Allowed: {allowed}")
        return v


class GuardrailOutputValidationService:

    def __init__(self) -> None:
        self._report_guard = Guard.for_pydantic(ReportOutput)
        self._analysis_guard = Guard.for_pydantic(AnalysisOutput)
        self._task_guard = Guard.for_pydantic(TaskAssignment)

    def validate_report_output(self, summary: str) -> tuple[bool, str]:
        result = self._report_guard.parse(json.dumps({"summary": summary}))
        if result.validation_passed:
            return True, "OK"
        return False, str(result.error)

    def validate_analysis_output(self, analysis: str) -> tuple[bool, str]:
        result = self._analysis_guard.parse(json.dumps({"analysis": analysis}))
        if result.validation_passed:
            return True, "OK"
        return False, str(result.error)

    def validate_task_assignment(self, tool_name: str) -> tuple[bool, str]:
        result = self._task_guard.parse(json.dumps({"tool_name": tool_name}))
        if result.validation_passed:
            return True, "OK"
        return False, str(result.error)

    def validate_tool_decision(self, tool_name: str, arguments: dict) -> tuple[bool, str]:
        ok, msg = self.validate_task_assignment(tool_name)
        if not ok:
            return False, msg

        required: dict[str, list[str]] = {
            "read_file": ["path"],
            "fetch_data": ["url"],
            "summarise_text": ["text"],
            "send_email": ["to", "subject", "body"],
        }
        needed = required.get(tool_name, [])
        for key in needed:
            val = arguments.get(key, "")
            if not isinstance(val, str) or not val.strip():
                return False, f"'{tool_name}' requires non-empty argument '{key}'"

        return True, "OK"
