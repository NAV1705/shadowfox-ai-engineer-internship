import os
from typing import List

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = (
    "You are a document question-answering assistant. Answer ONLY using the "
    "provided context excerpts, which were retrieved via semantic (embedding) "
    "similarity search. If the answer is not contained in the context, say so "
    "clearly rather than guessing. Cite excerpt numbers like [1] or [2]. Do "
    "not use outside knowledge."
)


def build_prompt(question: str, chunks: List[dict]) -> str:
    context_block = "\n\n".join(
        f"[{i + 1}] (source: {c['filename']}, chunk {c['chunk_id']}, "
        f"similarity: {c['score']:.2f})\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return (
        f"Context excerpts:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, citing excerpt numbers in brackets."
    )


def call_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
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
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def generate_answer(question: str, chunks: List[dict]) -> str:
    if not chunks:
        return "The uploaded documents do not contain information relevant to this question."
    prompt = build_prompt(question, chunks)
    if LLM_PROVIDER == "anthropic":
        return call_anthropic(prompt)
    if LLM_PROVIDER == "groq":
        return call_groq(prompt)
    return call_openai(prompt)
