# RAG-Lite Assistant — Production-Style Document Q&A (Keyword Retrieval, No Embeddings)

A production-style grounded question-answering system over PDF / TXT / Markdown
documents, built to match the ShadowFox AI Engineer internship **Advanced Level**
brief — with one deliberate substitution: retrieval uses **BM25 keyword search**
instead of embeddings + a vector database (no FAISS, no embedding model calls).

## Why keyword retrieval instead of embeddings?

BM25 is a classic sparse information-retrieval ranking function: it scores each
chunk by term frequency, inverse document frequency, and document length, with
no learned vector representation involved. It's fast, fully explainable (you can
see exactly why a chunk scored the way it did), and needs no embedding API calls
or vector index — at the cost of missing semantic/paraphrase matches that
embeddings would catch (e.g. matching "car" to "automobile"). For a task with a
bounded, keyword-rich document set, that trade-off is reasonable and keeps the
whole retrieval step local and dependency-light.

## Architecture

```
                 ┌──────────────┐        ┌───────────────────────────────┐
                 │  Streamlit    │  HTTP  │            FastAPI             │
                 │  frontend     │──────▶│  /documents/upload  /query      │
                 └──────────────┘        └───────────────┬─────────────────┘
                                                          │
                                          ┌───────────────▼───────────────┐
                                          │      LangGraph pipeline        │
                                          │  retrieve → rerank → generate  │
                                          │        → groundedness          │
                                          └───────────────┬───────────────┘
                                                          │
                     ┌────────────────────────────────────┼─────────────────┐
                     ▼                                    ▼                 ▼
             BM25 keyword index                    LLM (OpenAI /      Lexical
          (rank_bm25, in-memory)                     Anthropic)      overlap check
```

### Components

| Layer | File | Responsibility |
|---|---|---|
| Ingestion | `backend/ingestion.py` | Extract text from PDF/TXT/MD, clean whitespace, paragraph-aware chunking with overlap |
| Indexing / retrieval | `backend/retrieval.py` | Per-document BM25 index; keyword search + lexical reranking |
| Orchestration | `backend/workflow.py` | LangGraph state machine: `retrieve → rerank → generate → groundedness` |
| Generation | `backend/llm_client.py` | Builds a grounded prompt from retrieved chunks, calls the configured LLM, computes a groundedness score |
| API | `backend/main.py` | FastAPI routes, typed request/response validation, error handling |
| UI | `frontend/app.py` | Streamlit upload + query interface, shows retrieved context and confidence |

### Pipeline detail

1. **Ingestion** — `pypdf` for PDFs, plain decode for TXT/MD. Text is cleaned
   (line-ending/whitespace normalization) then chunked paragraph-by-paragraph
   into ~800-character blocks (configurable), splitting only oversized
   paragraphs, with a trailing overlap stitched onto each chunk so a sentence
   spanning a boundary isn't lost.
2. **Retrieval** — each document gets its own `BM25Okapi` index over tokenized
   chunks. A query is tokenized the same way and scored against every chunk in
   the selected document(s); top candidates are pooled across documents.
3. **Rerank** — a lightweight lexical pass on top of the BM25 shortlist:
   boosts chunks with higher exact query-term coverage and an exact phrase
   match, since BM25's raw term-frequency score can under-weight short,
   keyword-heavy questions.
4. **Generate** — the top reranked chunks are assembled into a numbered
   context block; the LLM is instructed (system prompt) to answer *only* from
   that context and cite excerpt numbers, or say the documents don't contain
   the answer.
5. **Groundedness check** — a token-overlap ratio between the answer and the
   retrieved context is computed as a cheap, explainable proxy for whether the
   answer is actually supported by the documents (no embeddings needed here
   either). Below a threshold, the UI flags the answer as low-confidence.

### How this addresses the brief's requirements

- **Multi-format ingestion**: PDF, TXT, Markdown.
- **Document-scoped retrieval**: queries are scoped to explicitly selected `doc_ids`.
- **Query refinement**: the reranking step re-weights BM25 candidates using
  exact-term coverage, functioning as a retrieval-quality refinement stage.
- **Groundedness / hallucination control**: system prompt constrains the model
  to context-only answers with citations, plus a post-hoc lexical groundedness
  score surfaced in the UI.
- **Typed validation**: all API I/O is defined with Pydantic models
  (`backend/models.py`), so malformed requests are rejected before reaching
  the pipeline.
- **Clean API structure**: ingestion, retrieval, generation, and orchestration
  are separate modules with a single FastAPI layer on top.
- **Containerized setup**: separate backend/frontend Dockerfiles plus a
  `docker-compose.yml` for one-command reproducibility.

## Running locally (without Docker)

```bash
cd rag_lite_assistant
python -m venv .venv && source .venv/bin/activate

pip install -r backend/requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY (or switch LLM_PROVIDER=anthropic)
uvicorn backend.main:app --reload --port 8000

# in a second terminal
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

Open the Streamlit URL it prints, upload a PDF/TXT/MD file, select it, and ask
a question.

## Running with Docker

```bash
cd rag_lite_assistant
cp .env.example .env   # fill in your API key
docker compose up --build
```

- Backend: http://localhost:8000/docs (interactive OpenAPI docs)
- Frontend: http://localhost:8501

## API reference

### `POST /documents/upload`
`multipart/form-data`, field `files` (one or more PDF/TXT/MD files). Returns a
`doc_id`, chunk count, and status per file.

### `POST /query`
```json
{
  "doc_ids": ["<doc_id>"],
  "question": "What does the document say about X?",
  "top_k": 4
}
```
Returns the answer, a groundedness score + confidence label, and the exact
retrieved chunks (with source filename and chunk index) used to generate it.

## Limitations & possible extensions

- BM25 is lexical only — synonyms/paraphrases the exact wording doesn't share
  won't be retrieved. Swapping `retrieval.py` for a FAISS + embeddings index
  (the original brief's suggestion) would address that directly.
- The document store is in-memory and resets on restart; a real deployment
  would persist chunks/indices (e.g. SQLite + a saved BM25 index, or a proper
  search engine like Elasticsearch/OpenSearch).
- Token-level streaming isn't wired up in the current `/query` endpoint but
  both OpenAI and Anthropic SDKs support `stream=True` — `llm_client.py` is
  structured so this can be added without touching retrieval/reranking.
