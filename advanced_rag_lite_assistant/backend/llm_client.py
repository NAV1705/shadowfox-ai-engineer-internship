from typing import List

from .config import settings
from .retrieval import tokenize

SYSTEM_PROMPT = (
    "You are a document question-answering assistant. Answer ONLY using the "
    "provided context excerpts. If the answer is not contained in the context, "
    "say clearly that the documents do not contain enough information. "
    "Cite which excerpt number(s) you used, like [1] or [2]. Do not use "
    "outside knowledge and do not invent facts."
)


def build_prompt(question: str, chunks: List[dict]) -> str:
    context_block = "\n\n".join(
        f"[{i + 1}] (source: {c['filename']}, chunk {c['chunk_id']})\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return (
        f"Context excerpts:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, citing excerpt numbers in brackets."
    )


def call_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def call_anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")

def call_groq(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    resp = client.chat.completions.create(
        model=settings.GROQ_MODEL,
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
    if settings.LLM_PROVIDER == "anthropic":
        return call_anthropic(prompt)
    if settings.LLM_PROVIDER == "groq":
        return call_groq(prompt)
    return call_openai(prompt)

def groundedness_score(answer: str, chunks: List[dict]) -> float:
    """Heuristic lexical-overlap groundedness check (no embeddings needed):
    fraction of the answer's tokens that also appear somewhere in the
    retrieved context. A cheap, explainable proxy for 'is this answer
    actually supported by the documents' — low overlap is a signal the
    model may have drifted into unsupported/hallucinated territory."""
    if not chunks:
        return 0.0
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 0.0
    context_tokens = set()
    for c in chunks:
        context_tokens |= set(tokenize(c["text"]))
    if not context_tokens:
        return 0.0
    overlap = answer_tokens & context_tokens
    return round(len(overlap) / len(answer_tokens), 3)


def confidence_label(score: float) -> str:
    if score >= settings.GROUNDEDNESS_HIGH_THRESHOLD:
        return "high"
    if score >= settings.GROUNDEDNESS_LOW_THRESHOLD:
        return "medium"
    return "low"
