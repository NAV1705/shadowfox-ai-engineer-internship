import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # "openai" or "anthropic" — swap providers without touching pipeline code
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    TOP_K_RETRIEVE: int = int(os.getenv("TOP_K_RETRIEVE", "8"))   # BM25 candidate pool
    TOP_K_RERANK: int = int(os.getenv("TOP_K_RERANK", "4"))       # final chunks sent to LLM

    GROUNDEDNESS_LOW_THRESHOLD: float = float(os.getenv("GROUNDEDNESS_LOW_THRESHOLD", "0.15"))
    GROUNDEDNESS_HIGH_THRESHOLD: float = float(os.getenv("GROUNDEDNESS_HIGH_THRESHOLD", "0.5"))


settings = Settings()
