# Document Q&A Assistant (Intermediate Level) — Embeddings + Vector Retrieval

A Streamlit app that answers questions grounded in uploaded PDF/TXT/Markdown
documents, using real embeddings and FAISS vector search — not keyword
matching.

## Pipeline

1. **Ingestion** (`ingestion.py`) — extracts text (`pypdf` for PDFs), cleans
   it, and splits it into ~800-character paragraph-aware chunks with a
   stitched overlap so context isn't lost at chunk boundaries.
2. **Embedding** (`embeddings.py`) — each chunk is converted into a dense
   vector. Two providers are supported, switchable via `EMBEDDING_PROVIDER`:
   - `local` (default): `sentence-transformers/all-MiniLM-L6-v2`, runs fully
     offline, no API key or per-call cost.
   - `openai`: OpenAI's `text-embedding-3-small` API.
3. **Indexing** (`vector_store.py`) — vectors are L2-normalized and stored in
   a FAISS `IndexFlatIP` index; normalized inner product is equivalent to
   cosine similarity, which is the standard way to do cosine search in FAISS.
4. **Retrieval** — a question is embedded with the same model, and FAISS
   returns the most semantically similar chunks (oversampled and filtered to
   the selected document(s), since a flat index has no native metadata
   filter).
5. **Grounded generation** (`qa.py`) — the retrieved chunks are numbered and
   inserted into a prompt that instructs the LLM to answer only from that
   context and cite excerpt numbers, or admit the documents don't contain the
   answer.
6. **UI** (`app.py`) — Streamlit handles upload, document selection, the
   question box, and displays the answer plus the exact retrieved chunks with
   their similarity scores, so the grounding is visible and checkable.

## Why embeddings here (vs. the advanced-level keyword version)

Embeddings capture *semantic* similarity — a question asking about "vehicles"
can retrieve a chunk that only says "car" or "truck", which keyword/BM25
search would miss. That's the specific capability this intermediate task is
meant to exercise: document ingestion → chunking → embeddings → vector
retrieval → grounded generation, as one working pipeline.

## Validation & error handling

- Empty files, unsupported file types, and documents with no extractable text
  are caught during ingestion and reported per-file without crashing the app.
- An empty question or no selected document is blocked before any embedding
  or LLM call is made.
- Embedding or generation failures (bad key, network error) are caught and
  shown as a readable error instead of a stack trace.

## Running it

```bash
cd intermediate_doc_qa
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # defaults to local embeddings; add OPENAI_API_KEY for generation
streamlit run app.py
```

The first local-embedding run will download the small `all-MiniLM-L6-v2`
model (~80MB) — this needs internet access once, then it's cached locally.

To use OpenAI embeddings instead of the local model, set
`EMBEDDING_PROVIDER=openai` in `.env` (uses the same `OPENAI_API_KEY` as
generation).
