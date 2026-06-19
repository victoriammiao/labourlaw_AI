# step3_search.py - 劳动法知识库语义检索测试

import sys

PROJECT_ROOT = "/root/autodl-tmp/projects/Labor Law Legal Advisor"
sys.path.insert(0, PROJECT_ROOT)

from langchain_chroma import Chroma

from config import COLLECTION_NAME, DB_PATH
from step2_embedding import get_embeddings


def load_vectorstore() -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )


def semantic_search(vectorstore: Chroma, query: str, top_k: int = 3) -> list[dict]:
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    output = []
    for i, (doc, distance) in enumerate(results, start=1):
        output.append(
            {
                "rank": i,
                "source": doc.metadata.get("source", "未知"),
                "content": doc.page_content,
                "distance": distance,
            }
        )
    return output


def print_search_result(query: str, results: list[dict]) -> None:
    print(f"\n{'=' * 65}")
    print(f"查询：{query}")
    print(f"{'=' * 65}")
    for r in results:
        print(f"\n  [第{r['rank']}名] 来源：{r['source']} | 距离：{r['distance']:.4f}")
        print(f"  内容：{r['content'][:120]}...")
    print()


if __name__ == "__main__":
    vectorstore = load_vectorstore()
    count = vectorstore._collection.count()
    print(f"劳动法知识库已加载，共 {count} 个文档块\n")

    if count == 0:
        print("⚠️  知识库为空，请先执行：")
        print("  python import_from_dify.py")
        print("  python step2_embedding.py")
        sys.exit(1)

    test_queries = [
        "试用期最长可以约定多久？",
        "公司拖欠工资怎么办？",
        "解除劳动合同需要赔偿吗？",
        "加班费如何计算？",
        "工伤认定需要哪些材料？",
    ]

    print("\n【劳动法检索测试】")
    for q in test_queries:
        print_search_result(q, semantic_search(vectorstore, q, top_k=2))

    print("\n✅ 检索测试完成")
