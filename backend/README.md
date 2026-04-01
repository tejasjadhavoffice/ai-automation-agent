# Week 1 Automation Agent

This backend implements Week 1 requirements:

- Groq model call with structured prompt
- JSON output parsing and Pydantic validation
- Tool routing and execution
- Retry with exponential backoff
- Graceful no-tool handling

## Setup

1. Activate virtual environment in `backend/venv`.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Create/update `backend/.env` with required keys:
   - `GROQ_API_KEY`
   - `SMTP_HOST`
   - `SMTP_PORT`
   - `SMTP_USER`
   - `SMTP_PASSWORD`
   - `EMAIL_FROM`

## Run

From `backend/`:

- Zero-shot:
  - `python -m app.main --prompt-style zero-shot --request "Read file sample.txt"`
- Few-shot:
  - `python -m app.main --prompt-style few-shot --request "Read file sample.txt"`
- Chain-of-thought:
  - `python -m app.main --prompt-style cot --request "Read file sample.txt"`
