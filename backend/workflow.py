from typing import List, TypedDict

from langgraph.graph import END, StateGraph

from . import llm_client
from .config import settings
from .retrieval import store


class RAGState(TypedDict):
    doc_ids: List[str]
    question: str
    top_k: int
    candidates: list
    reranked: list
    answer: str
    groundedness: float
    confidence: str


def retrieve_node(state: RAGState) -> RAGState:
    state["candidates"] = store.search(
        state["doc_ids"], state["question"], settings.TOP_K_RETRIEVE
    )
    return state


def rerank_node(state: RAGState) -> RAGState:
    state["reranked"] = store.rerank(
        state["question"], state["candidates"], state["top_k"]
    )
    return state


def generate_node(state: RAGState) -> RAGState:
    state["answer"] = llm_client.generate_answer(state["question"], state["reranked"])
    return state


def groundedness_node(state: RAGState) -> RAGState:
    score = llm_client.groundedness_score(state["answer"], state["reranked"])
    state["groundedness"] = score
    state["confidence"] = llm_client.confidence_label(score)
    return state


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("groundedness_check", groundedness_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "groundedness_check")
    graph.add_edge("groundedness_check", END)

    return graph.compile()


rag_graph = build_graph()


def run_pipeline(doc_ids: List[str], question: str, top_k: int) -> RAGState:
    initial_state: RAGState = {
        "doc_ids": doc_ids,
        "question": question,
        "top_k": top_k,
        "candidates": [],
        "reranked": [],
        "answer": "",
        "groundedness": 0.0,
        "confidence": "low",
    }
    return rag_graph.invoke(initial_state)