\# ShadowFox AI Engineer Internship — Task Submission



Three progressively harder AI applications, one per internship level. Each

folder is a standalone, runnable project with its own README, requirements,

and `.env.example`.



| Level | Folder | What it is |

|---|---|---|

| Beginner | \[`beginner\_student\_utility/`](./beginner\_student\_utility) | Flask web app — AI-powered student utility (summarize notes, generate quiz, improve an answer, explain a concept) |

| Intermediate | \[`intermediate\_doc\_qa/`](./intermediate\_doc\_qa) | Streamlit app — document Q\&A using real embeddings (local or OpenAI) + FAISS vector search |

| Advanced | \[`advanced\_rag\_lite\_assistant/`](./advanced\_rag\_lite\_assistant) | FastAPI + Streamlit + Docker — production-style RAG assistant, orchestrated with LangGraph, using BM25 keyword retrieval instead of embeddings |



\## Quick start (each level)



Every folder is self-contained:



cd <level-folder>

python -m venv .venv \&\& source .venv/bin/activate   # or .venv\\Scripts\\Activate.ps1 on Windows

pip install -r requirements.txt

cp .env.example .env    # add your API key (OpenAI, Anthropic, or Groq)



Then follow that folder's README for the exact run command (`python app.py`

for beginner, `streamlit run app.py` for intermediate, or the advanced

folder's own instructions for FastAPI + Streamlit + Docker).



\## Design notes across levels



\- \*\*Beginner\*\* focuses on prompt design and basic app plumbing: one Flask

&#x20; route, task-specific prompt templates, input validation, and error

&#x20; handling around a single LLM call.

\- \*\*Intermediate\*\* introduces a real retrieval pipeline: chunking →

&#x20; embeddings → vector index (FAISS) → similarity search → grounded

&#x20; generation.

\- \*\*Advanced\*\* turns that pipeline into a multi-step, orchestrated system

&#x20; (LangGraph: retrieve → rerank → generate → groundedness check) behind a

&#x20; typed FastAPI backend, containerized with Docker — retrieval here uses BM25

&#x20; keyword search rather than embeddings, a deliberate substitution explained

&#x20; in that folder's README.



All three levels have been built and tested end to end.

