import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG-Lite Assistant", layout="wide")
st.title("📄 RAG-Lite Document Assistant")
st.caption("Keyword-search (BM25) retrieval + LLM grounded answers — no embeddings, no vector DB.")

if "documents" not in st.session_state:
    st.session_state.documents = []  # list of {doc_id, filename, num_chunks, status}

with st.sidebar:
    st.header("1. Upload documents")
    uploaded = st.file_uploader(
        "PDF, TXT, or Markdown files", type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Upload") and uploaded:
        files = [("files", (f.name, f.getvalue())) for f in uploaded]
        try:
            resp = requests.post(f"{BACKEND_URL}/documents/upload", files=files, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            st.session_state.documents.extend(data["documents"])
            for d in data["documents"]:
                if d["status"] == "ready":
                    st.success(f"{d['filename']}: {d['num_chunks']} chunks indexed")
                else:
                    st.error(f"{d['filename']}: failed to process")
        except requests.exceptions.RequestException as e:
            st.error(f"Upload failed: {e}")

    st.divider()
    st.header("2. Select documents to query")
    ready_docs = [d for d in st.session_state.documents if d["status"] == "ready"]
    selected = st.multiselect(
        "Documents",
        options=[d["doc_id"] for d in ready_docs],
        format_func=lambda did: next(d["filename"] for d in ready_docs if d["doc_id"] == did),
    )

st.header("3. Ask a question")
question = st.text_input("Your question")
ask = st.button("Ask", type="primary")

if ask:
    if not question.strip():
        st.warning("Please enter a question.")
    elif not selected:
        st.warning("Please select at least one uploaded document.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"doc_ids": selected, "question": question},
                    timeout=60,
                )
                resp.raise_for_status()
                result = resp.json()

                st.subheader("Answer")
                st.write(result["answer"])

                badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}[result["confidence_label"]]
                st.caption(
                    f"{badge} Groundedness: {result['groundedness_score']:.2f} "
                    f"({result['confidence_label']} confidence)"
                )

                with st.expander("Show retrieved context"):
                    for i, c in enumerate(result["retrieved_chunks"], start=1):
                        st.markdown(
                            f"**[{i}] {c['filename']} — chunk {c['chunk_id']}** "
                            f"(score: {c['score']:.2f})"
                        )
                        st.text(c["text"])
                        st.divider()
            except requests.exceptions.RequestException as e:
                st.error(f"Query failed: {e}")
