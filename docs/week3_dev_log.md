# Week 3 Dev Log — Production Workflows

**Candidate:** Tejas | **Track:** AI Engineer
**Week theme:** Guardrails, Idempotency, Observability, Failure Recovery

---

## What was built

| File | What it is | Why it exists |
|---|---|---|
| `app/guardrails/guardrail_checker.py` | Validates LLM output before the agent acts | Prevents bad/empty LLM responses from causing side effects |
| `app/workflows/base_workflow.py` | Parent class for all workflows | Gives idempotency and file-saving to every workflow for free |
| `app/workflows/report_workflow.py` | Workflow 1 — report generation | Reads raw data → LLM report → saved to file |
| `app/workflows/analysis_workflow.py` | Workflow 2 — data analysis | Reads raw data → LLM trend bullets → saved to file |
| `app/workflows/task_scheduler_workflow.py` | Workflow 3 — task routing | Reads task list → LLM assigns tools → logs done tasks |
| `app/logging_setup.py` | Updated to add `log_step()` | Writes structured JSON lines to `logs/agent_steps.jsonl` |
| `app/automation_agent.py` | Added `run_workflow()` method | Entry point for all 3 workflows from the CLI |
| `app/main.py` | Added `--workflow` CLI flag | `python -m app.main --workflow report` |
| `data/sales_data.txt` | Sample raw sales data | Input for Workflow 1 and 2 |
| `data/tasks.txt` | Sample task list | Input for Workflow 3 |

---

## Key concepts explained

### 1. Guardrails — `app/guardrails/guardrail_checker.py`

**What:** A guardrail is a validation check that runs AFTER the LLM responds but BEFORE the agent does anything with the response.

**Why:** LLMs can return empty strings, garbage text, or hallucinated values. Without a guardrail, an empty report would be saved to disk and nobody would know.

**Which library:** `guardrails-ai` (v0.10.0) with Pydantic validators inside.

**How the code works:**

```python
# 1. Define what valid output looks like using Pydantic
class ReportOutput(BaseModel):
    summary: str

    @field_validator("summary")
    @classmethod
    def summary_long_enough(cls, v: str) -> str:
        if len(v.strip()) < 30:
            raise ValueError("Too short")  # this blocks the output
        return v

# 2. Wrap it in a Guard
guard = Guard.for_pydantic(ReportOutput)

# 3. In the workflow, call check before saving
ok, msg = self.checker.check_report(summary)
if not ok:
    return {"success": False, "message": f"Guardrail failed: {msg}"}
```

**Example of guardrail failing:**
If the LLM returns `""` (empty string), `check_report("")` returns `(False, "Too short")`.
The workflow stops and returns `success: False`. No file is written.

---

### 2. Idempotency — `_output_exists()` in `base_workflow.py`

**What:** A workflow is idempotent if running it twice gives the same result as running it once — no duplicates.

**Why:** In production, workflows are retried after failures. If your "send email" step runs twice, the user gets two emails. Idempotency prevents this.

**How the code works:**

```python
# In base_workflow.py
def _output_exists(self, filename: str) -> bool:
    return (self.output_dir / filename).exists()

# In report_workflow.py — checked FIRST before doing any work
if self._output_exists(output_file):
    return {"success": True, "message": "Already done — report exists"}
```

**Task scheduler idempotency:**
Instead of checking a file, Workflow 3 reads `data/processed_tasks.txt`.
Any task already in that file is skipped. After completing a task, it's appended to the file.

---

### 3. Structured Logging — `log_step()` in `logging_setup.py`

**What:** Every workflow step calls `log_step()` which writes one JSON line to `logs/agent_steps.jsonl`.

**Why:** Normal `print()` and `logging.info()` output readable text. Structured JSON logs can be:
- Searched by field (`"step": "guardrail"`)
- Loaded into dashboards (Datadog, Grafana)
- Used to trace exactly which step failed in production

**How the code works:**

```python
def log_step(workflow, step, input_data, output, decision):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "workflow": workflow,
        "step": step,
        "input": str(input_data)[:300],   # truncated so logs stay small
        "output": str(output)[:300],
        "decision": decision,
    }
    with open("logs/agent_steps.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")  # one JSON object per line
```

**Sample log entry:**
```json
{"timestamp": "2026-04-28T03:58:03.350686", "workflow": "ReportWorkflow", "step": "guardrail", "input": "Sales report...", "output": "True", "decision": "OK"}
```

---

### 4. Failure Simulations

Each workflow has one built-in failure simulation:

| Workflow | Failure | How it's handled |
|---|---|---|
| Report | `data/sales_data.txt` is missing | `FileNotFoundError` caught → returns `success: False` |
| Analysis | Data file has less than 10 characters | Explicit length check → returns `success: False` |
| Task Scheduler | LLM returns unknown tool name | Guardrail check fails → task skipped, others continue |

In all cases: no crash, no unhandled exception. The workflow logs the error and exits cleanly.

---

### 5. OOP Design — Why classes?

**`BaseWorkflow` (ABC — Abstract Base Class)**
- Contains shared code: idempotency check + file saving
- Forces every workflow to implement `run()` — Python raises `TypeError` if you forget

**`GuardrailChecker`**
- Groups all validation logic in one place
- Has one method per workflow (`check_report`, `check_analysis`, `check_task_assignment`)
- Easy to add new validators later without touching workflow code

**`AutomationAgent.run_workflow()`**
- Single entry point — the CLI just calls this with a name string
- Lazy imports inside the method — workflows only load when needed

---

## How to run

```bash
# Workflow 1 — generate report
python -m app.main --workflow report

# Workflow 2 — data analysis
python -m app.main --workflow analysis

# Workflow 3 — task scheduling
python -m app.main --workflow tasks

# Week 1 / Week 2 still work as before
python -m app.main --request "Read file data/sales_data.txt" --mode react
```

---

## File structure — Week 3 additions

```
app/
  guardrails/
    __init__.py
    guardrail_checker.py       ← guardrails-ai + Pydantic validators
  workflows/
    __init__.py
    base_workflow.py           ← idempotency + output saving (parent class)
    report_workflow.py         ← Workflow 1
    analysis_workflow.py       ← Workflow 2
    task_scheduler_workflow.py ← Workflow 3
  logging_setup.py             ← added log_step() for JSON file logging
  automation_agent.py          ← added run_workflow()
  main.py                      ← added --workflow CLI option

data/
  sales_data.txt               ← sample raw sales data
  tasks.txt                    ← sample task list
  processed_tasks.txt          ← auto-created; tracks done tasks (Workflow 3)
  reports/
    report_<date>.txt          ← auto-created by Workflow 1
    analysis_<date>.txt        ← auto-created by Workflow 2

logs/
  agent_steps.jsonl            ← auto-created; one JSON line per step

notebooks/
  week3_workflows.ipynb        ← Week 3 experiments notebook

docs/
  week3_dev_log.md             ← this file
```

---

## Week 3 evaluation checklist

| Requirement | Status | Where |
|---|---|---|
| 3 working workflows | ✅ | `app/workflows/` |
| Guardrail on every workflow | ✅ | `guardrail_checker.py` called in each workflow |
| Idempotency — no duplicate outputs | ✅ | `_output_exists()` in `BaseWorkflow`; `processed_tasks.txt` in Workflow 3 |
| Structured logging every step | ✅ | `log_step()` called at every step in all 3 workflows |
| One failure simulation per workflow | ✅ | Missing file / short data / invalid tool name |
| Failure handled gracefully | ✅ | All return `success: False` — no unhandled exceptions |
