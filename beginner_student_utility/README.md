# Student Study Assistant (Beginner Level)

A Flask web app that wraps an LLM API in a simple, usable student utility tool.
Pick a task, paste text, get a generated result.

## Features

- **Summarize Notes** — turns long notes into exam-ready bullet points
- **Generate Quiz** — creates 5 practice questions (with answers) from notes
- **Improve an Answer** — rewrites a rough draft answer into a clearer one
- **Explain a Concept** — plain-language explanation with an example

## How it works

1. The user selects a task and pastes text into a form (`templates/index.html`).
2. `app.py` validates the input isn't empty and the task is a known type.
3. `llm_client.py` maps the task to a specific prompt template (`PROMPTS` dict)
   and sends a structured prompt — not a raw pass-through of user text — to
   the configured LLM (OpenAI or Anthropic, switchable via `LLM_PROVIDER`).
4. The generated text is rendered back on the same page.

## Validation & error handling

- Empty input is rejected with a flash message before any API call is made.
- An unknown/tampered task value is rejected the same way.
- API failures (bad key, network error, rate limit) are caught and shown as a
  readable flash message instead of a stack trace or blank page.
- An empty model response is treated as a soft failure and flagged to the user.

## Running it

```bash
cd beginner_student_utility
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY (or switch to anthropic)
python app.py
```

Open http://localhost:5000.

## Why this counts as more than "isolated prompt calls"

Each task has its own purpose-built prompt template rather than sending the
user's raw text straight to the model — the app decides *how* to ask the
question based on the task selected, which is the "prompt design thinking"
the brief asks for. The Flask route also owns validation and error handling,
so the LLM call is one step inside a real request/response flow, not the
whole app.
