"""LangChain tools for legal document workflows."""

from __future__ import annotations

import os
import sys
from functools import lru_cache

PROJECT_ROOT = "/root/autodl-tmp/projects/Labor Law Legal Advisor"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import TAVILY_API_KEY, WEB_SEARCH_ENABLED
from langchain_core.tools import tool
from step3_search import load_vectorstore, semantic_search


@lru_cache(maxsize=1)
def _get_vectorstore():
    return load_vectorstore()


@tool
def labor_law_rag(query: str) -> str:
    """查询本地劳动法知识库，适合检索劳动合同法、劳动争议调解仲裁法、工伤保险条例。"""
    vectorstore = _get_vectorstore()
    results = semantic_search(vectorstore, query, top_k=5)
    if not results:
        return "本地劳动法知识库没有检索到相关资料。"

    sections = []
    for item in results:
        title = " ".join(
            part
            for part in (item.get("law_name"), item.get("article_no"), item.get("topic"))
            if part
        )
        sections.append(
            f"[{item['rank']}] {title or item.get('source', '未知来源')}\n"
            f"来源：{item.get('source', '')}\n"
            f"检索方式：{item.get('retrieval', '')}\n"
            f"内容：{item.get('content', '')}"
        )
    return "\n\n".join(sections)


def build_tools():
    """Build LangChain tools. Tavily is optional and controlled by .env."""
    tools = [labor_law_rag]
    if WEB_SEARCH_ENABLED:
        if not TAVILY_API_KEY:
            raise RuntimeError("WEB_SEARCH_ENABLED=true requires TAVILY_API_KEY in .env")
        os.environ.setdefault("TAVILY_API_KEY", TAVILY_API_KEY)
        try:
            from langchain_community.tools import TavilySearchResults
        except ImportError as exc:
            raise RuntimeError(
                "TavilySearchResults requires langchain-community and tavily-python."
            ) from exc
        tools.append(TavilySearchResults(max_results=5))
    return tools
