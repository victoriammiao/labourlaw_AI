# step2_embedding.py - 将本地知识库切片向量化并存入 ChromaDB

import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import BGE_MODEL_NAME, BGE_MODEL_PATH, COLLECTION_NAME, DB_PATH
from data.rag_law.chunking import build_documents


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the BGE embedding model, preferring the configured local cache."""
    model_name = BGE_MODEL_PATH if os.path.exists(BGE_MODEL_PATH) else BGE_MODEL_NAME
    print(f"正在加载 Embedding 模型：{model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore() -> Chroma:
    """Rebuild the Chroma vector store from structured labor-law documents."""
    chunks = build_documents()
    if not chunks:
        raise RuntimeError("没有可向量化的切片，请检查 data/rag_law/imported/knowledge.json")

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(f"已删除旧向量库：{DB_PATH}")

    embeddings = get_embeddings()
    print(f"\n开始向量化并入库，共 {len(chunks)} 个文档块...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
    )
    print(f"✅ 入库完成，共 {vectorstore._collection.count()} 条记录")
    return vectorstore


if __name__ == "__main__":
    vectorstore = build_vectorstore()

    print("\n── 数据库概览 ──")
    sample = vectorstore._collection.get(limit=3, include=["documents", "metadatas"])
    for i, doc_id in enumerate(sample["ids"]):
        print(f"  ID: {doc_id}")
        print(f"  来源: {sample['metadatas'][i].get('source', '未知')}")
        print(f"  内容: {sample['documents'][i][:80]}...")
        print()
