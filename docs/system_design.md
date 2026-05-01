# System Design Document — AI Automation Agent
**Candidate:** Tejas | **Track:** AI Engineer | **Week:** 4

---

## 1. Architecture Overview

The agent is built as a **ReAct loop** (Reason → Act → Observe) on top of a Groq-hosted LLM (Llama 3.1).

```
User request (CLI)
      │
      ▼
AgentCli (main.py)
      │
      ▼
AutomationAgent  ──► GroqChatClient  ──► Groq API (LLM)
      │
      ▼
AgentOrchestrationService.run_week4()
      │
      ├── Ambiguity check  ──► stop + print clarifying question
      ├── GuardrailChecker ──► block bad tool calls before execution
      ├── ToolExecutionService ──► execute approved tool
      └── log_step() ──► logs/agent_steps.jsonl (trace_id per run)
```

**Why this architecture?**
A pipeline (step 1 → step 2 → step 3 hardcoded) cannot handle variable-length tasks. The ReAct loop lets the LLM decide dynamically how many steps are needed and which tools to use. It also makes it easy to add guardrails at a single boundary point.

---

## 2. Why ReAct over alternatives

| Alternative | Problem |
|---|---|
| Hardcoded pipeline | Cannot handle variable tasks — must rewrite for every new request |
| Single LLM call | Cannot chain tools — one response cannot read a file AND summarise it |
| ReAct loop (chosen) | LLM reasons at each step, one tool per step, observations feed back in |

**Trade-off accepted:** ReAct uses more LLM tokens (multiple round trips vs one). For a small agent this is fine. At scale, caching common subtasks would reduce cost.

---

## 3. How tools are designed

Each tool is a plain function in `app/tools/`. The `ToolExecutionService` routes by name.

```
read_file      → reads a local file, returns content
fetch_data     → HTTP GET a URL, returns JSON or text
summarise_text → calls LLM to summarise text passed in
send_email     → SMTP send via configured credentials
```

**When to use a tool vs reason directly:**
If the answer requires external data (file, API, computation), use a tool.
If the answer can be derived from what is already in the context window, the LLM reasons directly and sets `done: true`.

---

## 4. Guardrails

A guardrail runs **after** the LLM responds but **before** the agent acts.

```python
ok, msg = checker.check_tool_decision(tool_name, arguments)
if not ok:
    return error   # stop — never execute
```

What is checked:
- `tool_name` is one of the 5 allowed values (not a hallucinated name)
- Required arguments for that tool are present and non-empty (e.g. `path` for `read_file`)

**Example of a guardrail failing without this:**
LLM returns `tool_name: "web_search"` (not a real tool). Without the guardrail the code crashes with `KeyError`. With the guardrail it logs the block and returns `success: False`.

---

## 5. Observability

Every agent step writes one JSON line to `logs/agent_steps.jsonl`:

```json
{"timestamp": "...", "trace_id": "a1b2c3", "workflow": "Week4Agent", "step": "step_1_tool_read_file", "input": "...", "output": "...", "decision": "True"}
```

`trace_id` is a 6-character random hex ID generated at the start of each run.
This lets you filter all lines for one run: search `"trace_id": "a1b2c3"`.

**To diagnose a broken agent from its logs:**
1. Find the `trace_id` from the failed run in console output.
2. Filter `agent_steps.jsonl` by that `trace_id`.
3. Read step by step — find where `decision` is `"error"` or `"False"` or `"guardrail_blocked"`.
4. Check the `input` and `output` fields on that line to see what went wrong.

---

## 6. Memory management

**Short-term memory (context window):**
The agent keeps a list of `memory_lines` — one per completed step. These are joined and passed back to the LLM as "Prior steps" in the next prompt. They are trimmed to `max_memory_chars = 4500` characters so the context window does not overflow.

**Long-term memory:**
Not implemented (Week 4 scope). Would use a vector store (e.g. ChromaDB or Pinecone) to store summaries of past runs so the agent can reference them in future sessions.

---

## 7. Scaling to concurrent requests

Currently the agent is single-threaded (one request at a time via CLI).

To handle concurrent requests in production:
- Wrap `AutomationAgent.run_week4()` in a **FastAPI** endpoint (one endpoint per request, each gets its own `trace_id`).
- Run behind **Gunicorn + Uvicorn** workers — each worker handles one request independently.
- The Groq API already handles rate limits via the retry logic in `GroqChatClient`.
- `logs/agent_steps.jsonl` should move to a structured log aggregator (Datadog, CloudWatch) rather than a local file, so all workers write to the same sink.

---

## 8. Monitoring and alerting in production

| What to monitor | How |
|---|---|
| LLM call latency | Log `elapsed_ms` on each `groq.complete_chat()` call |
| Guardrail block rate | Count log lines where `decision = "guardrail_blocked"` — spike = LLM degrading |
| `max_steps` hit rate | Count log lines where `stopped = "max_steps"` — spike = tasks getting stuck |
| Tool error rate | Count log lines where tool `success = False` |
| API error rate | Groq retry logs — `Groq retry attempt=3` means hitting rate limits |

Alert when: guardrail block rate > 5%, or `max_steps` rate > 10% of runs.

---

## 9. Handling LLM provider outages

`GroqChatClient` already has:
- 3 retries with exponential backoff (1s → 2s → 4s)
- Retry on: 429 rate limit, 5xx server errors, timeouts

Additional options for production:
- **Fallback provider:** If Groq is down, switch to OpenAI or Anthropic using the same `complete_chat()` interface — just swap the client.
- **Circuit breaker:** After N consecutive failures, stop sending requests and return a degraded response immediately (avoids piling up timeouts).
- **Cache common responses:** For deterministic requests (same input → same output), cache in Redis so the agent can respond even when the LLM is unavailable.

---

## 10. Guardrails before deploying to real users

Before going to real users, add:
1. **Input sanitisation** — reject requests that contain file paths outside allowed directories (path traversal attack).
2. **Output content filter** — check LLM output for personally identifiable information (PII) before logging or emailing.
3. **Rate limiting per user** — prevent one user from exhausting the Groq quota.
4. **Human-in-the-loop for email/send actions** — require explicit confirmation before `send_email` executes (the agent should ask "shall I send this?" and wait for `yes`).
5. **Audit log** — immutable record of every action taken (what was sent, to whom, when) for compliance.
