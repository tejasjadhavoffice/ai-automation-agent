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

    thought: str = ""
    subtasks: list[str] = Field(default_factory=list)
    done: bool = False
    tool_name: AllowedToolName
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    # Week 4: agent signals ambiguity instead of guessing
    needs_clarification: bool = False
    clarifying_question: str = ""

    @model_validator(mode="after")
    def validate_step(self) -> "ReactStep":
        # If the LLM picks no_tool without setting done=True and without a reason,
        # it most likely means it thinks it's finished — treat it as done defensively.
        if self.tool_name == "no_tool" and not self.done and not self.needs_clarification and not self.reason.strip():
            self.done = True
        return self
