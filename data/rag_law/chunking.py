"""Build structured article-level chunks from the local legal knowledge JSON."""

from __future__ import annotations

import json
import os
import re
import sys

from langchain_core.documents import Document

PROJECT_ROOT = "/root/autodl-tmp/projects/Labor Law Legal Advisor"
sys.path.insert(0, PROJECT_ROOT)

from config import IMPORTED_DATA_PATH


ARTICLE_RE = re.compile(r"^(第[一二三四五六七八九十百千万零〇两0-9]+条)\s*(.*)")
CHAPTER_RE = re.compile(r"^(第[一二三四五六七八九十百千万零〇两0-9]+章)\s*(.*)")
SECTION_RE = re.compile(r"^(第[一二三四五六七八九十百千万零〇两0-9]+节)\s*(.*)")


def load_imported_payload(path: str = IMPORTED_DATA_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到本地知识库文件：{path}\n"
            "请先准备 data/rag_law/imported/knowledge.json，"
            "或直接使用已有 ChromaDB 向量库运行 RAG API。"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _law_name_from_source(source: str) -> str:
    name = os.path.basename(source)
    name = re.sub(r"\.(docx?|pdf|txt|md)$", "", name, flags=re.I)
    name = re.sub(r"[_-]?\d{6,8}$", "", name)
    if name == "劳动合同法":
        return "中华人民共和国劳动合同法"
    return name


def _is_noise(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    if not normalized:
        return True
    if normalized in {"目录", "目次"}:
        return True
    return len(normalized) <= 2


def _topic_from_text(article_no: str, article_text: str) -> str:
    body = article_text.replace(article_no, "", 1).strip()
    body = re.split(r"[。；;：:，,\n]", body, maxsplit=1)[0].strip()
    return (body or article_no)[:40]


def _format_article_content(
    law_name: str,
    article_no: str,
    chapter: str,
    section: str,
    topic: str,
    article_text: str,
) -> str:
    parts = [
        f"法律名称：{law_name}",
        f"条号：{article_no}",
    ]
    if chapter:
        parts.append(f"章：{chapter}")
    if section:
        parts.append(f"节：{section}")
    if topic and topic != article_no:
        parts.append(f"主题：{topic}")
    parts.extend(["条文原文：", article_text.strip()])
    return "\n".join(parts)


def _build_article_documents_for_source(source: str, chunks: list[dict]) -> list[Document]:
    documents: list[Document] = []
    law_name = _law_name_from_source(source)
    chapter = ""
    section = ""
    current: dict | None = None

    def flush_current() -> None:
        nonlocal current
        if not current:
            return

        article_text = "\n".join(current["parts"]).strip()
        topic = _topic_from_text(current["article_no"], article_text)
        metadata = {
            "source": source,
            "law_name": law_name,
            "article_no": current["article_no"],
            "chapter": current["chapter"],
            "section": current["section"],
            "topic": topic,
            "start_position": current["start_position"],
            "end_position": current["end_position"],
            "chunk_type": "article",
        }
        if current.get("document_id"):
            metadata["document_id"] = current["document_id"]

        documents.append(
            Document(
                page_content=_format_article_content(
                    law_name=law_name,
                    article_no=current["article_no"],
                    chapter=current["chapter"],
                    section=current["section"],
                    topic=topic,
                    article_text=article_text,
                ),
                metadata=metadata,
            )
        )
        current = None

    for item in sorted(chunks, key=lambda c: c.get("metadata", {}).get("position", 0)):
        text = item.get("content", "").strip()
        metadata = item.get("metadata", {})
        position = metadata.get("position", 0)
        if _is_noise(text):
            continue

        if CHAPTER_RE.match(text):
            chapter = text
            section = ""
            continue
        if SECTION_RE.match(text):
            section = text
            continue

        article_match = ARTICLE_RE.match(text)
        if article_match:
            flush_current()
            current = {
                "article_no": article_match.group(1),
                "chapter": chapter,
                "section": section,
                "start_position": position,
                "end_position": position,
                "document_id": metadata.get("document_id", ""),
                "parts": [text],
            }
            continue

        if current:
            current["parts"].append(text)
            current["end_position"] = position

    flush_current()
    return documents


def build_documents(path: str = IMPORTED_DATA_PATH) -> list[Document]:
    """Return one LangChain Document per legal article with structured metadata."""
    payload = load_imported_payload(path)
    chunks_by_source: dict[str, list[dict]] = {}
    for item in payload.get("chunks", []):
        source = item.get("metadata", {}).get("source", "未知")
        chunks_by_source.setdefault(source, []).append(item)

    documents: list[Document] = []
    for source, chunks in chunks_by_source.items():
        documents.extend(_build_article_documents_for_source(source, chunks))

    if documents:
        return documents

    # Fallback for unexpected input formats.
    fallback: list[Document] = []
    for item in payload.get("chunks", []):
        content = item.get("content", "").strip()
        if not content or _is_noise(content):
            continue
        metadata = dict(item.get("metadata", {}))
        metadata.setdefault("chunk_type", "raw")
        fallback.append(Document(page_content=content, metadata=metadata))
    return fallback


def show_chunk_stats(path: str = IMPORTED_DATA_PATH) -> None:
    payload = load_imported_payload(path)
    documents = build_documents(path)
    by_source: dict[str, int] = {}
    for item in documents:
        source = item.metadata.get("source", "未知")
        by_source[source] = by_source.get(source, 0) + 1

    print(f"\n知识库 dataset_id: {payload.get('dataset_id')}")
    print(f"文档数: {payload.get('document_count', 0)}")
    print(f"原始切片数: {payload.get('chunk_count', len(payload.get('chunks', [])))}")
    print(f"结构化条文 chunk 数: {len(documents)}")
    print("\n各文档条文统计：")
    for source, count in by_source.items():
        print(f"  {source}: {count} 条")


if __name__ == "__main__":
    show_chunk_stats()
    docs = build_documents()
    if docs:
        sample = docs[0]
        print(f"\n示例 metadata: {sample.metadata}")
        print(f"\n示例内容:\n{sample.page_content[:500]}")
