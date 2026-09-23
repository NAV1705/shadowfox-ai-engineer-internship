import os

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = (
    "You are a helpful study assistant for students. Be clear, accurate, and "
    "concise. Format your output for easy reading (use bullet points or "
    "numbered lists where appropriate)."
)

PROMPTS = {
    "summarize": (
        "Summarize the following student notes into clear, concise bullet "
        "points that capture the key ideas. Keep it easy to review before an "
        "exam.\n\n{content}"
    ),
    "quiz": (
        "Based on the following notes, generate 5 short quiz questions (a mix "
        "of multiple choice and short answer) with the answers listed at the "
        "end, to help a student test their understanding.\n\n{content}"
    ),
    "improve_answer": (
        "The following is a student's draft answer to a question. Rewrite it "
        "into a clearer, more complete, and better-structured answer, while "
        "keeping the original meaning and any correct points intact.\n\n{content}"
    ),
    "explain": (
        "Explain the following concept or topic in simple, student-friendly "
        "language, with a short example if it helps understanding.\n\n{content}"
    ),
}


def build_prompt(task: str, content: str) -> str:
    template = PROMPTS.get(task)
    if not template:
        raise ValueError(f"Unknown task type: {task}")
    return template.format(content=content.strip())


def call_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content or ""


def call_anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def call_groq(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content or ""


def run_task(task: str, content: str) -> str:
    prompt = build_prompt(task, content)
    if LLM_PROVIDER == "anthropic":
        return call_anthropic(prompt)
    if LLM_PROVIDER == "groq":
        return call_groq(prompt)
    return call_openai(prompt)
