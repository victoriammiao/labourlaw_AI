"""Build prompt context from user-uploaded attachments."""

from __future__ import annotations

import re


TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]+")
MAX_ATTACHMENT_CONTEXT_CHARS = 6000

DOC_GENERATION_RE = re.compile(
    r"起草|撰写|生成|写一份|写个|立案申请|起诉状|仲裁申请|答辩状|证据目录|律师函"
)
ATTACHMENT_QUERY_RE = re.compile(
    r"文件|附件|上传|材料|文档|合同|协议|通知书|条款|上面|这份|该份|"
    r"说了什么|讲了什么|内容是什么|什么意思|帮我看|解读|摘要|总结|概括",
    re.I,
)


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text or ""))


def _score_chunk(query_tokens: set[str], chunk: str) -> float:
    chunk_tokens = _tokenize(chunk)
    if not query_tokens or not chunk_tokens:
        return 0.0
    return len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)


def is_document_generation_request(query: str) -> bool:
    return bool(DOC_GENERATION_RE.search(query or ""))


def is_attachment_focused_query(query: str) -> bool:
    return bool(ATTACHMENT_QUERY_RE.search(query or ""))


def format_planner_attachment_hint(attachments: list[dict] | None) -> str:
    if not attachments:
        return "（当前对话无上传文件）"
    lines = ["当前对话已附加以下文件（内容已解析入库，无需让用户重新上传）："]
    for item in attachments:
        lines.append(f"- {item.get('name', '未知文件')}（约 {item.get('chars', 0)} 字）")
    return "\n".join(lines)


def should_use_attachment_first_path(query: str, attachment_context: str) -> bool:
    if not attachment_context:
        return False
    if is_document_generation_request(query):
        return False
    return is_attachment_focused_query(query)


def select_attachment_context(attachments: list[dict] | None, query: str) -> str:
    if not attachments:
        return ""

    total_chars = sum(item.get("chars", len(item.get("content", ""))) for item in attachments)
    if total_chars <= MAX_ATTACHMENT_CONTEXT_CHARS:
        sections = []
        used = 0
        for item in attachments:
            block = f"【文件：{item.get('name', '未知文件')}】\n{item.get('content', '').strip()}"
            if used and used + len(block) > MAX_ATTACHMENT_CONTEXT_CHARS:
                remaining = MAX_ATTACHMENT_CONTEXT_CHARS - used - 16
                if remaining > 200:
                    block = block[:remaining] + "\n...(内容过长，已截断)..."
                else:
                    break
            sections.append(block)
            used += len(block) + 2
        return "\n\n".join(sections)

    query_tokens = _tokenize(query)
    scored_chunks: list[tuple[float, str, str]] = []
    for item in attachments:
        name = item.get("name", "未知文件")
        chunks = re.split(r"\n{2,}", item.get("content", ""))
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            scored_chunks.append((_score_chunk(query_tokens, chunk), name, chunk))

    scored_chunks.sort(key=lambda row: row[0], reverse=True)
    sections = []
    used = 0
    for _, name, chunk in scored_chunks:
        block = f"【文件：{name}】\n{chunk}"
        if used + len(block) > MAX_ATTACHMENT_CONTEXT_CHARS:
            break
        sections.append(block)
        used += len(block)

    if not sections:
        first = attachments[0]
        fallback = first.get("content", "")[:MAX_ATTACHMENT_CONTEXT_CHARS]
        return f"【文件：{first.get('name', '未知文件')}】\n{fallback}"

    return "\n\n".join(sections)


def format_attachment_status(attachments: list[dict] | None) -> str:
    from web.document_parser import MAX_FILES_PER_SESSION, MAX_SESSION_ATTACHMENT_CHARS

    if not attachments:
        return "当前对话暂无上传文件。（每个对话最多 5 个文件，总提取字数上限约 45000）"

    total_chars = sum(item.get("chars", 0) for item in attachments)
    lines = [
        f"**已附加文件（{len(attachments)}/{MAX_FILES_PER_SESSION}，共约 {total_chars}/{MAX_SESSION_ATTACHMENT_CHARS} 字）：**",
    ]
    for item in attachments:
        lines.append(f"- {item.get('name', '未知文件')}（{item.get('chars', 0)} 字）")
    lines.append("\n提问时会结合这些文件内容回答。列表里出现几个，就是当前对话里实际保存了几个。")
    return "\n".join(lines)


def format_attachment_reference(attachments: list[dict] | None) -> str:
    if not attachments:
        return ""
    names = "、".join(item.get("name", "未知文件") for item in attachments)
    return f"\n\n---\n\n> 本次回答参考了上传文件：{names}"
