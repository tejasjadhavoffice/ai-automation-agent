"""
guardrail_checker.py

Uses the guardrails-ai library (Guard.for_pydantic) to validate LLM output
before the agent acts on it. Each Pydantic model defines what "valid output" looks like.
If the output fails the check, the workflow stops and logs the reason.
"""

import json
from typing import Literal

from guardrails import Guard
from pydantic import BaseModel, field_validator


# --- Constants for guardrail thresholds ---

MIN_REPORT_LENGTH = 30
MIN_ANALYSIS_LENGTH = 20

AllowedToolName = Literal[
    "read_file", "send_email", "fetch_data", "summarise_text", "no_tool"
]


# --- Pydantic models define what valid LLM output looks like ---

class ReportOutput(BaseModel):
    """Valid output from the report-generation step."""
    summary: str

    @field_validator("summary")
    @classmethod
    def summary_long_enough(cls, v: str) -> str:
        if len(v.strip()) < MIN_REPORT_LENGTH:
            raise ValueError(f"Report summary too short (minimum {MIN_REPORT_LENGTH} characters)")
        return v


class AnalysisOutput(BaseModel):
    """Valid output from the data-analysis step."""
    analysis: str

    @field_validator("analysis")
    @classmethod
    def analysis_not_empty(cls, v: str) -> str:
        if len(v.strip()) < MIN_ANALYSIS_LENGTH:
            raise ValueError(f"Analysis output too short (minimum {MIN_ANALYSIS_LENGTH} characters)")
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


class GuardrailOutputValidationService:
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

    def validate_report_output(self, summary: str) -> tuple[bool, str]:
        """Validate that a report summary is long enough to be useful."""
        result = self._report_guard.parse(json.dumps({"summary": summary}))
        if result.validation_passed:
            return True, "OK"
        return False, str(result.error)

    def validate_analysis_output(self, analysis: str) -> tuple[bool, str]:
        """Validate that an analysis output is non-trivial."""
        result = self._analysis_guard.parse(json.dumps({"analysis": analysis}))
        if result.validation_passed:
            return True, "OK"
        return False, str(result.error)

    def validate_task_assignment(self, tool_name: str) -> tuple[bool, str]:
        """Validate that the LLM assigned a real, allowed tool name."""
        result = self._task_guard.parse(json.dumps({"tool_name": tool_name}))
        if result.validation_passed:
            return True, "OK"
        return False, str(result.error)

    def validate_tool_decision(self, tool_name: str, arguments: dict) -> tuple[bool, str]:
        """
        Week 4 guardrail — called at every output-to-action boundary in run_week4().

        Checks two things:
          1. tool_name is one of the allowed values (reuses TaskAssignment model).
          2. Required arguments for that tool are present and non-empty.

        Returns (True, "OK") if valid, (False, reason) if not.
        """
        # Step 1: validate tool name
        ok, msg = self.validate_task_assignment(tool_name)
        if not ok:
            return False, msg

        # Step 2: validate required arguments per tool
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
