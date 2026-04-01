from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

AllowedToolName = Literal[
    "read_file", "send_email", "fetch_data", "summarise_text", "no_tool"
]


class LlmToolDecision(BaseModel):
    """Validated instruction returned by the LLM."""

    tool_name: AllowedToolName
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""

    @model_validator(mode="after")
    def validate_no_tool_reason(self) -> "LlmToolDecision":
        if self.tool_name == "no_tool" and not self.reason.strip():
            raise ValueError("reason is required when tool_name is no_tool")
        return self
