"""加载本地知识库切片（来自 Dify 导入的 JSON）。"""

from __future__ import annotations

import json
import os
import sys

from langchain_core.documents import Document

PROJECT_ROOT = "/root/autodl-tmp/projects/Labor Law Legal Advisor"
sys.path.insert(0, PROJECT_ROOT)

from config import IMPORTED_DATA_PATH


def load_imported_payload(path: str = IMPORTED_DATA_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到本地知识库文件：{path}\n"
            "请先执行：\n"
            "  export DIFY_API_KEY='dataset-xxx'\n"
            "  export DIFY_DATASET_ID='知识库UUID'\n"
            "  python import_from_dify.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_documents(path: str = IMPORTED_DATA_PATH) -> list[Document]:
    """将 Dify 导入的切片转为 LangChain Document 列表。"""
    payload = load_imported_payload(path)
    documents: list[Document] = []

    for item in payload.get("chunks", []):
        content = item.get("content", "").strip()
        if not content:
            continue
        metadata = dict(item.get("metadata", {}))
        documents.append(Document(page_content=content, metadata=metadata))

    return documents


def show_chunk_stats(path: str = IMPORTED_DATA_PATH) -> None:
    payload = load_imported_payload(path)
    chunks = payload.get("chunks", [])
    by_source: dict[str, int] = {}
    for item in chunks:
        source = item.get("metadata", {}).get("source", "未知")
        by_source[source] = by_source.get(source, 0) + 1

    print(f"\n知识库 dataset_id: {payload.get('dataset_id')}")
    print(f"文档数: {payload.get('document_count', 0)}")
    print(f"切片数: {payload.get('chunk_count', len(chunks))}")
    print("\n各文档切片统计：")
    for source, count in by_source.items():
        print(f"  {source}: {count} 块")


if __name__ == "__main__":
    show_chunk_stats()
    docs = build_documents()
    print(f"\n可加载 Document 数量: {len(docs)}")
    if docs:
        sample = docs[0]
        print(f"\n示例切片（来源: {sample.metadata.get('source')}）：")
        print(sample.page_content[:200])
