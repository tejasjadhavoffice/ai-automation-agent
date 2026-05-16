import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.automation_agent import AutomationAgent
from app.logging_setup import setup_console_logging

logger = logging.getLogger(__name__)

class AgentRequest(BaseModel):
    request: str = Field(
        ...,
        min_length=1,
        description="What you want the agent to do",
        examples=["Read data/sales_data.txt and summarise it"],
    )
    prompt_style: str = Field(
        default="zero-shot",
        description="Prompt engineering style: zero-shot, few-shot, or cot",
        examples=["zero-shot"],
    )


class AgentResponse(BaseModel):

    success: bool
    message: str
    data: dict | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

_agent: AutomationAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global _agent
    setup_console_logging()
    logger.info("step=api_startup decision=initializing_agent")
    _agent = AutomationAgent.initialize()
    logger.info("step=api_startup decision=agent_ready")
    yield  
    logger.info("step=api_shutdown decision=cleanup_done")


app = FastAPI(
    title="AI Automation Agent API",
    description="REST API for the AI Automation Agent — supports single tool calls, "
                "multi-step ReAct reasoning, and automated workflows.",
    version="1.0.0",
    lifespan=lifespan,
)



def _get_agent() -> AutomationAgent:

    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail="Agent is not initialized yet. Server is still starting up.",
        )
    return _agent



@app.get("/health", tags=["System"])
def health_check():

    logger.info("step=health_check decision=ok")
    return {"status": "healthy", "message": "Agent API is running"}


@app.post("/api/v1/agent/once", response_model=AgentResponse, tags=["Agent"])
def run_agent_once(body: AgentRequest):
    agent = _get_agent()
    logger.info(
        "step=api_once decision=received request_len=%d prompt_style=%s",
        len(body.request), body.prompt_style,
    )

    try:
        result = agent.execute(
            user_request=body.request,
            mode="once",
            prompt_style=body.prompt_style,
        )
        logger.info("step=api_once decision=success")
        return AgentResponse(success=True, message="Single tool executed", data=result)

    except Exception as exc:
        logger.error("step=api_once decision=error err=%s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")


@app.post("/api/v1/agent/react", response_model=AgentResponse, tags=["Agent"])
def run_agent_react(body: AgentRequest):

    agent = _get_agent()
    logger.info(
        "step=api_react decision=received request_len=%d",
        len(body.request),
    )

    try:
        result = agent.execute(
            user_request=body.request,
            mode="react",
        )
        logger.info("step=api_react decision=success")
        return AgentResponse(success=True, message="ReAct loop completed", data=result)

    except Exception as exc:
        logger.error("step=api_react decision=error err=%s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")


@app.post("/api/v1/workflows/{workflow_name}", response_model=AgentResponse, tags=["Workflows"])
def run_workflow(workflow_name: str):
    agent = _get_agent()

    allowed_workflows = ["report", "analysis", "tasks"]
    if workflow_name not in allowed_workflows:
        logger.warning(
            "step=api_workflow decision=invalid_name workflow=%s",
            workflow_name,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Unknown workflow: '{workflow_name}'. Choose from: {allowed_workflows}",
        )

    logger.info("step=api_workflow decision=received workflow=%s", workflow_name)

    try:
        result = agent.execute_workflow(workflow_name)
        logger.info("step=api_workflow decision=success workflow=%s", workflow_name)
        return AgentResponse(
            success=True,
            message=f"Workflow '{workflow_name}' completed",
            data=result,
        )

    except Exception as exc:
        logger.error(
            "step=api_workflow decision=error workflow=%s err=%s",
            workflow_name, exc, exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Workflow error: {exc}")
