"""HTTP client helpers for the local RAG API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


RERANK_RELEVANCE_THRESHOLD = 0.3

OUT_OF_KB_KEYWORDS = (
    "婚姻法",
    "离婚",
    "财产分割",
    "继承",
    "刑法",
    "犯罪",
    "罪名",
    "判刑",
    "公司法",
    "知识产权",
    "专利",
    "商标",
    "著作权",
    "行政法",
    "行政许可",
    "治安管理处罚",
)


def ask_rag_api(query: str, api_url: str, top_k: int) -> dict | None:
    payload = json.dumps({"query": query, "top_k": top_k}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"RAG API unavailable: {exc}")
        return None


def has_evidence(rag_data: dict | None) -> bool:
    return bool(rag_data and rag_data.get("results") and rag_data.get("context"))


def is_out_of_knowledge_scope(query: str) -> bool:
    return any(keyword in query for keyword in OUT_OF_KB_KEYWORDS)


def has_relevant_evidence(rag_data: dict | None, query: str = "") -> bool:
    if not has_evidence(rag_data):
        return False

    top = rag_data["results"][0]
    if top.get("retrieval") == "exact":
        return True

    score = top.get("rerank_score")
    if score is None:
        return True

    return float(score) >= RERANK_RELEVANCE_THRESHOLD


def build_no_evidence_message(query: str) -> str:
    if is_out_of_knowledge_scope(query):
        return (
            "本地劳动法知识库仅收录《劳动合同法》《劳动争议调解仲裁法》《工伤保险条例》，"
            "未检索到与该问题相关的法条资料。"
        )
    return (
        "这次没有检索到可用的劳动法资料。你可以补充具体场景、涉及的合同条款、"
        "工资支付情况或争议经过，我再帮你继续查。"
    )


def build_boundary_notice(query: str) -> str:
    if not is_out_of_knowledge_scope(query):
        return ""
    return (
        "> **知识库说明**：本地知识库未收录该领域法规（如婚姻法、刑法等），"
        "以下回答来自联网搜索，仅供参考。\n\n"
    )


def format_reference_markdown(rag_data: dict | None, max_items: int = 3, max_chars: int = 180) -> str:
    if not rag_data or not has_relevant_evidence(rag_data):
        return ""

    lines = [
        "",
        "---",
        "",
        '<div class="reference-card">',
        "",
        "#### 参考资料",
        "",
    ]
    for item in rag_data.get("results", [])[:max_items]:
        content = item.get("content", "").strip().replace("\n", " ")
        if len(content) > max_chars:
            content = f"{content[:max_chars]}..."
        title_parts = [item.get("law_name", ""), item.get("article_no", ""), item.get("topic", "")]
        title = " ".join(part for part in title_parts if part) or item.get("source", "未知来源")
        score_text = ""
        if item.get("rerank_score") is not None:
            score_text = f" · rerank {item['rerank_score']:.3f}"
        lines.extend(
            [
                f"**[{item.get('rank')}] {title}**",
                f"`{item.get('source', '未知来源')}` · `{item.get('retrieval', 'vector')}`{score_text}",
                "",
                f"> {content}",
                "",
            ]
        )
    lines.append("</div>")
    return "\n".join(lines)


def format_rag_only_answer(decision, rag_data: dict | None) -> str:
    if not has_relevant_evidence(rag_data):
        return "未检索到可用的劳动法资料。请确认 RAG API 已启动，或换一种更具体的问法。"

    lines = [
        f"工作流意图：{decision.intent}",
        f"检索问题：{decision.rewritten_query}",
        "",
        "以下是本地劳动法知识库检索到的相关依据：",
        "",
    ]
    for item in rag_data.get("results", []):
        lines.extend(
            [
                f"[{item['rank']}] 来源：{item['source']}，距离：{item['distance']:.4f}",
                item["content"],
                "",
            ]
        )
    lines.append("说明：当前为 RAG-only 模式，仅展示检索依据；配置 Qwen 模型后可生成完整回答。")
    return "\n".join(lines)
