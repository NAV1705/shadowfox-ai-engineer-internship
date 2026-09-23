from typing import Dict, List, Optional

import faiss
import numpy as np


class VectorStore:
    """Thin wrapper around a FAISS flat inner-product index. Vectors are
    L2-normalized on insert and query, so inner product is equivalent to
    cosine similarity — a standard trick for cosine search with FAISS."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict] = []

    def add(self, vectors: np.ndarray, metadatas: List[Dict]) -> None:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.metadata.extend(metadatas)

    def search(
        self, query_vector: np.ndarray, top_k: int, doc_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        if self.index.ntotal == 0:
            return []

        query_vector = np.ascontiguousarray(query_vector.reshape(1, -1), dtype="float32")
        faiss.normalize_L2(query_vector)

        # Flat FAISS indices have no native metadata filter, so oversample
        # and filter by doc_id in Python, then trim to top_k.
        k = min(self.index.ntotal, max(top_k * 5, top_k)) if doc_ids else min(self.index.ntotal, top_k)
        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            if doc_ids and meta["doc_id"] not in doc_ids:
                continue
            results.append({**meta, "score": float(score)})
            if len(results) >= top_k:
                break
        return results
