from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

AllowedToolName = Literal[
    "read_file", "send_email", "fetch_data", "summarise_text", "no_tool"
]


class LlmToolDecision(BaseModel):

    tool_name: AllowedToolName
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""

    @model_validator(mode="after")
    def validate_no_tool_reason(self) -> "LlmToolDecision":
        if self.tool_name == "no_tool" and not self.reason.strip():
            raise ValueError("reason is required when tool_name is no_tool")
        return self

class ReactStep(BaseModel):

    thought: str = ""          # LLM sometimes omits this — default to empty string
    subtasks: list[str] = Field(default_factory=list)
    done: bool = False
    tool_name: AllowedToolName
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""

    @model_validator(mode="after")
    def validate_step(self) -> "ReactStep":
        if not self.done and self.tool_name == "no_tool" and not self.reason.strip():
            raise ValueError("no_tool requires reason when done is false")
        return self
