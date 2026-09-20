import re
from dataclasses import dataclass, field
from typing import Dict, List

from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


@dataclass
class DocumentIndex:
    filename: str
    chunks: List[str]
    tokenized: List[List[str]] = field(init=False)
    bm25: BM25Okapi = field(init=False)

    def __post_init__(self):
        self.tokenized = [tokenize(c) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized)


class RetrievalStore:
    """In-memory keyword-search index. Uses BM25 (a classic sparse IR ranking
    function) over tokenized chunks instead of dense embeddings + a vector DB.
    This keeps retrieval fully explainable (scores are term-frequency based)
    at the cost of not capturing semantic/paraphrase similarity the way
    embeddings would."""

    def __init__(self):
        self._docs: Dict[str, DocumentIndex] = {}

    def add_document(self, doc_id: str, filename: str, chunks: List[str]) -> None:
        self._docs[doc_id] = DocumentIndex(filename=filename, chunks=chunks)

    def has(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def search(self, doc_ids: List[str], query: str, top_k: int) -> List[dict]:
        query_tokens = tokenize(query)
        candidates: List[dict] = []
        for doc_id in doc_ids:
            doc = self._docs.get(doc_id)
            if doc is None:
                continue
            scores = doc.bm25.get_scores(query_tokens)
            for idx, score in enumerate(scores):
                if score > 0:
                    candidates.append({
                        "doc_id": doc_id,
                        "filename": doc.filename,
                        "chunk_id": idx,
                        "text": doc.chunks[idx],
                        "score": float(score),
                    })
        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates[:top_k]

    def rerank(self, query: str, candidates: List[dict], top_k: int) -> List[dict]:
        """Lexical rerank on top of the BM25 shortlist: rewards chunks with
        higher exact query-term coverage and an exact phrase match, which
        BM25's term-frequency scoring alone can under-weight for short,
        keyword-heavy questions."""
        query_terms = set(tokenize(query))
        if not query_terms:
            return candidates[:top_k]

        rescored = []
        for c in candidates:
            chunk_term_set = set(tokenize(c["text"]))
            coverage = len(query_terms & chunk_term_set) / len(query_terms)
            phrase_bonus = 1.0 if query.lower() in c["text"].lower() else 0.0
            combined = c["score"] * 0.6 + coverage * 5.0 + phrase_bonus * 5.0
            rescored.append({**c, "rerank_score": combined})

        rescored.sort(key=lambda c: c["rerank_score"], reverse=True)
        return rescored[:top_k]


store = RetrievalStore()
