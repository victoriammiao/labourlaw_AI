"""HTTP API for the local labor law RAG retriever."""

from __future__ import annotations

import sys
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import PROJECT_ROOT

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from step3_search import load_vectorstore, semantic_search


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(3, ge=1, le=10, description="Number of chunks to retrieve")


class SearchHit(BaseModel):
    rank: int
    source: str
    law_name: str = ""
    article_no: str = ""
    topic: str = ""
    chapter: str = ""
    section: str = ""
    content: str
    distance: float
    retrieval: str = ""
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    top_k: int
    count: int
    results: list[SearchHit]


class AskResponse(SearchResponse):
    context: str


app = FastAPI(
    title="Labor Law RAG API",
    description="Local retrieval API for the labor law RAG frontend.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_vectorstore():
    """Load the Chroma vector store once and reuse it across API requests."""
    return load_vectorstore()


def format_context(results: list[dict]) -> str:
    """Convert retrieved law chunks into the compact context passed to the UI/agent."""
    sections = []
    for item in results:
        title_parts = [item.get("law_name", ""), item.get("article_no", ""), item.get("topic", "")]
        title = " ".join(part for part in title_parts if part) or item["source"]
        rerank_line = ""
        if item.get("rerank_score") is not None:
            rerank_line = f"重排分数：{item['rerank_score']:.4f}\n"
        sections.append(
            f"[{item['rank']}] {title}\n"
            f"来源：{item['source']}\n"
            f"检索方式：{item.get('retrieval', 'vector')}\n"
            f"{rerank_line}"
            f"相关度距离：{item['distance']:.4f}\n"
            f"内容：{item['content']}"
        )
    return "\n\n".join(sections)


@app.get("/health")
def health() -> dict:
    """Report whether the local vector store is available."""
    try:
        vectorstore = get_vectorstore()
        count = vectorstore._collection.count()
    except Exception as exc:  # pragma: no cover - health should surface runtime failures
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", "vector_count": count}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Return ranked labor-law chunks for a user query."""
    vectorstore = get_vectorstore()
    results = semantic_search(vectorstore, request.query, top_k=request.top_k)
    return SearchResponse(
        query=request.query,
        top_k=request.top_k,
        count=len(results),
        results=[SearchHit(**item) for item in results],
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: SearchRequest) -> AskResponse:
    """Return search hits plus a formatted context block for answer generation."""
    vectorstore = get_vectorstore()
    results = semantic_search(vectorstore, request.query, top_k=request.top_k)
    return AskResponse(
        query=request.query,
        top_k=request.top_k,
        count=len(results),
        results=[SearchHit(**item) for item in results],
        context=format_context(results),
    )
