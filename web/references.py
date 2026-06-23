"""Build user-visible reference sections from actually used sources."""

from __future__ import annotations

from dataclasses import dataclass
import json

from web.rag_client import format_reference_markdown, has_relevant_evidence


@dataclass
class UsedSources:
    rag_data: dict | None = None
    web_results: list[dict] | None = None
    attachments: list[dict] | None = None


def format_attachment_reference_card(attachments: list[dict] | None, max_items: int = 5) -> str:
    if not attachments:
        return ""

    lines = [
        "",
        "---",
        "",
        '<div class="reference-card">',
        "",
        "#### 上传文件",
        "",
    ]
    for idx, item in enumerate(attachments[:max_items], start=1):
        name = item.get("name", "未知文件")
        chars = item.get("chars", len(item.get("content", "")))
        lines.extend(
            [
                f"**[{idx}] {name}**",
                f"已从该文件解析约 {chars} 字文本参与本次回答",
                "",
            ]
        )
    lines.append("</div>")
    return "\n".join(lines)


def format_web_reference_markdown(web_results: list[dict] | None, max_items: int = 5, max_chars: int = 180) -> str:
    if not web_results:
        return ""

    lines = [
        "",
        "---",
        "",
        '<div class="reference-card">',
        "",
        "#### 网页来源",
        "",
    ]
    for idx, item in enumerate(web_results[:max_items], start=1):
        content = str(item.get("content", "")).strip().replace("\n", " ")
        if len(content) > max_chars:
            content = f"{content[:max_chars]}..."
        title = item.get("title") or item.get("url") or "网页结果"
        url = item.get("url", "")
        lines.extend(
            [
                f"**[{idx}] {title}**",
                f"`{url}`" if url else "",
                "",
                f"> {content}" if content else "",
                "",
            ]
        )
    lines.append("</div>")
    return "\n".join(lines)


def format_used_sources(sources: UsedSources | None) -> str:
    if not sources:
        return ""

    parts: list[str] = []
    if sources.rag_data and has_relevant_evidence(sources.rag_data):
        parts.append(format_reference_markdown(sources.rag_data))
    if sources.web_results:
        parts.append(format_web_reference_markdown(sources.web_results))
    if sources.attachments:
        parts.append(format_attachment_reference_card(sources.attachments))
    return "".join(parts)


def _parse_tavily_content(content) -> list[dict]:
    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)]
    if not content:
        return []
    text = content if isinstance(content, str) else str(content)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return [item for item in data["results"] if isinstance(item, dict)]
    return []


def extract_agent_sources(raw: dict | None, *, rag_api_url: str, rag_top_k: int) -> UsedSources:
    if not raw:
        return UsedSources()

    try:
        from langchain_core.messages import AIMessage, ToolMessage
    except ImportError:
        return UsedSources()

    web_results: list[dict] = []
    rag_queries: list[str] = []

    for message in raw.get("messages", []):
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls or []:
                name = tool_call.get("name", "")
                args = tool_call.get("args") or {}
                if name == "labor_law_rag" and isinstance(args, dict):
                    query = str(args.get("query") or "").strip()
                    if query:
                        rag_queries.append(query)
        if isinstance(message, ToolMessage):
            name = (message.name or "").lower()
            if "tavily" in name:
                web_results.extend(_parse_tavily_content(message.content))

    rag_data = None
    if rag_queries:
        from web.rag_client import ask_rag_api

        rag_data = ask_rag_api(rag_queries[-1], rag_api_url, rag_top_k)
        if not has_relevant_evidence(rag_data):
            rag_data = None

    return UsedSources(
        rag_data=rag_data,
        web_results=web_results or None,
    )
