import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from embeddings import EMBEDDING_PROVIDER, embed_query, embed_texts, embedding_dim
from ingestion import process_document
from qa import generate_answer
from vector_store import VectorStore


st.set_page_config(page_title="Document Q&A (Embeddings)", layout="wide")
st.title("📚 Document Q&A Assistant — Embeddings + Vector Retrieval")
st.caption(
    f"Embedding provider: **{EMBEDDING_PROVIDER}** · FAISS vector search · "
    "grounded LLM answers"
)

if "store" not in st.session_state:
    st.session_state.store = VectorStore(dim=embedding_dim())
if "documents" not in st.session_state:
    st.session_state.documents = []  # {doc_id, filename, num_chunks}

with st.sidebar:
    st.header("1. Upload documents")
    uploaded = st.file_uploader(
        "PDF, TXT, or Markdown", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if st.button("Process & index") and uploaded:
        for f in uploaded:
            try:
                content = f.getvalue()
                if not content:
                    st.error(f"{f.name}: empty file")
                    continue
                doc_id, chunks = process_document(f.name, content)
                with st.spinner(f"Embedding {f.name} ({len(chunks)} chunks)..."):
                    vectors = embed_texts(chunks)
                metadatas = [
                    {"doc_id": doc_id, "filename": f.name, "chunk_id": i, "text": c}
                    for i, c in enumerate(chunks)
                ]
                st.session_state.store.add(np.array(vectors), metadatas)
                st.session_state.documents.append(
                    {"doc_id": doc_id, "filename": f.name, "num_chunks": len(chunks)}
                )
                st.success(f"{f.name}: indexed {len(chunks)} chunks")
            except Exception as e:
                st.error(f"{f.name}: failed — {e}")

    st.divider()
    st.header("2. Select documents")
    selected = st.multiselect(
        "Documents",
        options=[d["doc_id"] for d in st.session_state.documents],
        format_func=lambda did: next(
            d["filename"] for d in st.session_state.documents if d["doc_id"] == did
        ),
    )

st.header("3. Ask a question")
question = st.text_input("Your question")
top_k = st.slider("Chunks to retrieve", 1, 10, 4)
ask = st.button("Ask", type="primary")

if ask:
    if not question.strip():
        st.warning("Please enter a question.")
    elif not selected:
        st.warning("Please select at least one document.")
    else:
        try:
            with st.spinner("Embedding query and retrieving relevant context..."):
                q_vec = embed_query(question)
                results = st.session_state.store.search(np.array(q_vec), top_k, doc_ids=selected)

            if not results:
                st.warning("No relevant content found in the selected documents.")
            else:
                with st.spinner("Generating grounded answer..."):
                    answer = generate_answer(question, results)

                st.subheader("Answer")
                st.write(answer)

                with st.expander("Show retrieved context (with similarity scores)"):
                    for i, c in enumerate(results, start=1):
                        st.markdown(
                            f"**[{i}] {c['filename']} — chunk {c['chunk_id']}** "
                            f"(similarity: {c['score']:.3f})"
                        )
                        st.text(c["text"])
                        st.divider()
        except Exception as e:
            st.error(f"Query failed: {e}")
