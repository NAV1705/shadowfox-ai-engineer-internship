import os
from typing import List

import numpy as np

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")  # "openai" or "local"
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

EMBEDDING_DIMS = {"local": 384, "openai": 1536}

_local_model = None


def embedding_dim() -> int:
    return EMBEDDING_DIMS.get(EMBEDDING_PROVIDER, 384)


def _get_local_model():
    """Loads a small local sentence-transformer model on first use, so
    embeddings work fully offline with no API key or per-call cost."""
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_model


def embed_texts(texts: List[str]) -> np.ndarray:
    if EMBEDDING_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
        vectors = [d.embedding for d in resp.data]
        return np.array(vectors, dtype="float32")

    model = _get_local_model()
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.astype("float32")


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
