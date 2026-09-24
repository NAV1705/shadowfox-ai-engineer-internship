import io
import re
import uuid
from typing import List, Tuple

from pypdf import PdfReader

from .config import settings

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".markdown")


def extract_text(filename: str, content: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    if lower.endswith((".txt", ".md", ".markdown")):
        return content.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {filename}")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Paragraph-aware chunking: packs whole paragraphs into ~chunk_size blocks,
    hard-splitting only paragraphs that exceed chunk_size on their own, then
    stitches a small overlap onto the front of each chunk (after the first)
    so context isn't lost at chunk boundaries."""
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP
    if overlap >= chunk_size:
        overlap = chunk_size // 4

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    raw_chunks: List[str] = []
    buffer = ""

    for para in paragraphs:
        if len(buffer) + len(para) + 1 <= chunk_size:
            buffer = f"{buffer}\n{para}".strip()
            continue

        if buffer:
            raw_chunks.append(buffer)
            buffer = ""

        if len(para) > chunk_size:
            start = 0
            while start < len(para):
                end = start + chunk_size
                raw_chunks.append(para[start:end])
                start = end - overlap
        else:
            buffer = para

    if buffer:
        raw_chunks.append(buffer)

    if not raw_chunks:
        return [text] if text.strip() else []

    overlapped: List[str] = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_tail = raw_chunks[i - 1][-overlap:] if overlap > 0 else ""
        overlapped.append((prev_tail + "\n" + raw_chunks[i]).strip())

    return overlapped


def process_document(filename: str, content: bytes) -> Tuple[str, List[str]]:
    raw_text = extract_text(filename, content)
    if not raw_text or not raw_text.strip():
        raise ValueError(f"No extractable text found in {filename}")
    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned)
    if not chunks:
        raise ValueError(f"Document produced no usable chunks: {filename}")
    doc_id = str(uuid.uuid4())
    return doc_id, chunks
