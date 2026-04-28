"""
guardrail_checker.py

Uses the guardrails-ai library (Guard.for_pydantic) to validate LLM output
before the agent acts on it. Each Pydantic model defines what "valid output" looks like.
If the output fails the check, the workflow stops and logs the reason.
"""

import json

from guardrails import Guard
from pydantic import BaseModel, field_validator


# --- Pydantic models define what valid LLM output looks like ---

class ReportOutput(BaseModel):
    """Valid output from the report-generation step."""
    summary: str

    @field_validator("summary")
    @classmethod
    def summary_long_enough(cls, v: str) -> str:
        if len(v.strip()) < 30:
            raise ValueError("Report summary too short (minimum 30 characters)")
        return v


class AnalysisOutput(BaseModel):
    """Valid output from the data-analysis step."""
    analysis: str

    @field_validator("analysis")
    @classmethod
    def analysis_not_empty(cls, v: str) -> str:
        if len(v.strip()) < 20:
            raise ValueError("Analysis output too short (minimum 20 characters)")
        return v


class TaskAssignment(BaseModel):
    """Valid tool assignment returned by the task-scheduler LLM."""
    tool_name: str

    @field_validator("tool_name")
    @classmethod
    def must_be_valid_tool(cls, v: str) -> str:
        allowed = {"read_file", "fetch_data", "summarise_text", "send_email", "no_tool"}
        if v not in allowed:
            raise ValueError(f"'{v}' is not a valid tool. Allowed: {allowed}")
        return v


class GuardrailChecker:
    """
    Validates LLM outputs using guardrails-ai before the agent acts on them.

    How it works:
      1. Each workflow calls one of the check_* methods.
      2. guard.parse() wraps the value in JSON and runs Pydantic validators.
      3. Returns (True, "OK") or (False, error_message).
    """

    def __init__(self) -> None:
        self._report_guard = Guard.for_pydantic(ReportOutput)
        self._analysis_guard = Guard.for_pydantic(AnalysisOutput)
        self._task_guard = Guard.for_pydantic(TaskAssignment)

    def check_report(self, summary: str) -> tuple[bool, str]:
        """Validate that a report summary is long enough to be useful."""
        result = self._report_guard.parse(json.dumps({"summary": summary}))
        if result.validation_passed:
            return True, "OK"
        return False, str(result.error)

    def check_analysis(self, analysis: str) -> tuple[bool, str]:
        """Validate that an analysis output is non-trivial."""
        result = self._analysis_guard.parse(json.dumps({"analysis": analysis}))
        if result.validation_passed:
            return True, "OK"
        return False, str(result.error)

    def check_task_assignment(self, tool_name: str) -> tuple[bool, str]:
        """Validate that the LLM assigned a real, allowed tool name."""
        result = self._task_guard.parse(json.dumps({"tool_name": tool_name}))
        if result.validation_passed:
            return True, "OK"
        return False, str(result.error)
