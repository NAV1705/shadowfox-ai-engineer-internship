import logging
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .ingestion import SUPPORTED_EXTENSIONS, process_document
from .models import (
    DocumentInfo,
    DocumentStatus,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    UploadResponse,
)
from .retrieval import store
from .workflow import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-lite")

app = FastAPI(
    title="RAG-Lite Assistant (Keyword Retrieval, No Embeddings)",
    description=(
        "Production-style document Q&A assistant. Retrieval uses BM25 keyword "
        "search instead of embeddings/vector search, orchestrated as a "
        "multi-step LangGraph pipeline: retrieve -> rerank -> generate -> "
        "groundedness check."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    results: List[DocumentInfo] = []

    for f in files:
        if not f.filename.lower().endswith(SUPPORTED_EXTENSIONS):
            results.append(DocumentInfo(
                doc_id="", filename=f.filename, num_chunks=0,
                status=DocumentStatus.failed,
            ))
            continue
        try:
            content = await f.read()
            if not content:
                raise ValueError("Empty file")
            doc_id, chunks = process_document(f.filename, content)
            store.add_document(doc_id, f.filename, chunks)
            results.append(DocumentInfo(
                doc_id=doc_id, filename=f.filename,
                num_chunks=len(chunks), status=DocumentStatus.ready,
            ))
        except Exception:
            logger.exception("Failed to process %s", f.filename)
            results.append(DocumentInfo(
                doc_id="", filename=f.filename, num_chunks=0,
                status=DocumentStatus.failed,
            ))

    if all(r.status == DocumentStatus.failed for r in results):
        raise HTTPException(
            status_code=422,
            detail="None of the uploaded files could be processed. "
                   "Supported types: PDF, TXT, MD.",
        )

    return UploadResponse(documents=results)


@app.post("/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    valid_doc_ids = [d for d in payload.doc_ids if store.has(d)]
    if not valid_doc_ids:
        raise HTTPException(
            status_code=404,
            detail="None of the given doc_ids were found. Upload documents first.",
        )

    top_k = payload.top_k or settings.TOP_K_RERANK

    try:
        result = run_pipeline(valid_doc_ids, payload.question, top_k)
    except Exception as e:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {e}")

    retrieved = [
        RetrievedChunk(
            doc_id=c["doc_id"],
            filename=c["filename"],
            chunk_id=c["chunk_id"],
            text=c["text"],
            score=c.get("rerank_score", c["score"]),
        )
        for c in result["reranked"]
    ]

    return QueryResponse(
        question=payload.question,
        answer=result["answer"],
        groundedness_score=result["groundedness"],
        confidence_label=result["confidence"],
        retrieved_chunks=retrieved,
    )
