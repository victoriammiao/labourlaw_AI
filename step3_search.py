# step3_search.py - 劳动法知识库检索

import re
import os
import sys
from collections import Counter
from functools import lru_cache
from math import log

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_chroma import Chroma

from config import COLLECTION_NAME, DB_PATH, RERANKER_MODEL_NAME, RERANKER_MODEL_PATH
from step2_embedding import get_embeddings


LAW_ALIASES = (
    ("劳动争议调解仲裁法", "中华人民共和国劳动争议调解仲裁法"),
    ("调解仲裁法", "中华人民共和国劳动争议调解仲裁法"),
    ("劳动合同法", "中华人民共和国劳动合同法"),
    ("工伤保险条例", "工伤保险条例"),
)
ARTICLE_QUERY_RE = re.compile(
    r"(?P<law>劳动争议调解仲裁法|调解仲裁法|劳动合同法|工伤保险条例)?"
    r".*?(?:第)?(?P<num>[一二三四五六七八九十百零〇两0-9]+)条"
)
CN_DIGITS = "零一二三四五六七八九"
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]+")
STOPWORDS = {
    "什么",
    "怎么",
    "怎么办",
    "如何",
    "可以",
    "需要",
    "哪些",
    "一下",
    "具体",
    "规定",
    "处理",
}
CANDIDATE_K = 10


def load_vectorstore() -> Chroma:
    """Open the persisted Chroma collection used by the local RAG service."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )


def _law_name_from_query(query: str) -> str:
    for alias, law_name in LAW_ALIASES:
        if alias in query:
            return law_name
    return ""


def _int_to_chinese(num: int) -> str:
    if num <= 10:
        return "十" if num == 10 else CN_DIGITS[num]
    if num < 20:
        return f"十{CN_DIGITS[num % 10]}" if num % 10 else "十"
    if num < 100:
        tens, ones = divmod(num, 10)
        return f"{CN_DIGITS[tens]}十{CN_DIGITS[ones] if ones else ''}"
    hundreds, rest = divmod(num, 100)
    if rest == 0:
        return f"{CN_DIGITS[hundreds]}百"
    if rest < 10:
        return f"{CN_DIGITS[hundreds]}百零{CN_DIGITS[rest]}"
    return f"{CN_DIGITS[hundreds]}百{_int_to_chinese(rest)}"


def _normalize_article_no(raw: str) -> str:
    raw = raw.replace("〇", "零").replace("两", "二")
    if raw.isdigit():
        return f"第{_int_to_chinese(int(raw))}条"
    if raw.startswith("第") and raw.endswith("条"):
        return raw
    return f"第{raw}条"


def _extract_article_query(query: str) -> tuple[str, str]:
    match = ARTICLE_QUERY_RE.search(query)
    if not match:
        return "", ""
    law_name = _law_name_from_query(match.group("law") or query)
    article_no = _normalize_article_no(match.group("num"))
    return law_name, article_no


def _dict_from_hit(rank: int, content: str, metadata: dict, distance: float, retrieval: str = "vector") -> dict:
    return {
        "rank": rank,
        "source": metadata.get("source", "未知"),
        "law_name": metadata.get("law_name", ""),
        "article_no": metadata.get("article_no", ""),
        "topic": metadata.get("topic", ""),
        "chapter": metadata.get("chapter", ""),
        "section": metadata.get("section", ""),
        "content": content,
        "distance": distance,
        "retrieval": retrieval,
    }


def exact_article_search(vectorstore: Chroma, query: str) -> list[dict]:
    """Find an explicitly referenced law article before falling back to semantic search."""
    law_name, article_no = _extract_article_query(query)
    if not article_no:
        return []

    raw = vectorstore._collection.get(include=["documents", "metadatas"])
    matches = []
    for content, metadata in zip(raw.get("documents", []), raw.get("metadatas", []), strict=False):
        if metadata.get("article_no") != article_no:
            continue
        if law_name and metadata.get("law_name") != law_name:
            continue
        matches.append(_dict_from_hit(len(matches) + 1, content, metadata, 0.0, retrieval="exact"))
    return matches


def _tokenize(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", text)
    tokens = [token for token in TOKEN_RE.findall(text) if token not in STOPWORDS]
    # Character bigrams work well enough for short Chinese legal terms without adding jieba.
    tokens.extend(compact[i : i + 2] for i in range(max(0, len(compact) - 1)))
    return [token for token in tokens if token and token not in STOPWORDS]


def keyword_search(vectorstore: Chroma, query: str, top_k: int = 5) -> list[dict]:
    """Run a lightweight BM25-style keyword search over law names, article numbers, and text."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    raw = vectorstore._collection.get(include=["documents", "metadatas"])
    documents = raw.get("documents", [])
    metadatas = raw.get("metadatas", [])
    if not documents:
        return []

    tokenized_docs = []
    doc_freq = Counter()
    for content, metadata in zip(documents, metadatas, strict=False):
        weighted_text = " ".join(
            [
                metadata.get("law_name", ""),
                metadata.get("article_no", ""),
                metadata.get("topic", ""),
                content,
            ]
        )
        tokens = _tokenize(weighted_text)
        tokenized_docs.append(tokens)
        doc_freq.update(set(tokens))

    avg_len = sum(len(tokens) for tokens in tokenized_docs) / max(len(tokenized_docs), 1)
    n_docs = len(tokenized_docs)
    k1 = 1.5
    b = 0.75
    scored = []

    for idx, tokens in enumerate(tokenized_docs):
        counts = Counter(tokens)
        doc_len = len(tokens) or 1
        score = 0.0
        for token in query_tokens:
            tf = counts.get(token, 0)
            if not tf:
                continue
            idf = log((n_docs - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5) + 1)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_len))
        if score > 0:
            scored.append((score, idx))

    scored.sort(reverse=True)
    output = []
    for score, idx in scored[:top_k]:
        output.append(
            _dict_from_hit(
                len(output) + 1,
                documents[idx],
                metadatas[idx],
                max(0.0, 1.0 / (score + 1.0)),
                retrieval="keyword",
            )
        )
    return output


def _result_key(item: dict) -> tuple[str, str, str]:
    return (item.get("law_name", ""), item.get("article_no", ""), item.get("source", ""))


@lru_cache(maxsize=1)
def get_reranker():
    """Load the cross-encoder reranker lazily; retrieval still works if it is unavailable."""
    try:
        from sentence_transformers import CrossEncoder

        model_name = RERANKER_MODEL_PATH if os.path.exists(RERANKER_MODEL_PATH) else RERANKER_MODEL_NAME
        print(f"正在加载 Reranker 模型：{model_name}")
        return CrossEncoder(model_name, device="cpu")
    except Exception as exc:
        print(f"Reranker unavailable, fallback to hybrid retrieval: {exc}")
        return None


def _rerank_results(query: str, results: list[dict], top_k: int) -> list[dict]:
    if not results:
        return []

    exact_results = [item for item in results if item.get("retrieval") == "exact"]
    rest = [item for item in results if item.get("retrieval") != "exact"]
    if not rest:
        return exact_results[:top_k]

    reranker = get_reranker()
    if reranker is None:
        output = results[:top_k]
        for rank, item in enumerate(output, start=1):
            item["rank"] = rank
        return output

    scores = reranker.predict([(query, item["content"]) for item in rest])
    reranked = []
    for item, score in zip(rest, scores, strict=False):
        copied = dict(item)
        copied["rerank_score"] = float(score)
        copied["retrieval"] = f"{copied.get('retrieval', 'hybrid')}+rerank"
        reranked.append(copied)
    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

    output = [*exact_results, *reranked][:top_k]
    for rank, item in enumerate(output, start=1):
        item["rank"] = rank
    return output


def semantic_search(vectorstore: Chroma, query: str, top_k: int = 3, candidate_k: int = CANDIDATE_K) -> list[dict]:
    """Combine exact article, keyword, vector, RRF, and reranker results into final hits."""
    exact_results = exact_article_search(vectorstore, query)
    if exact_results:
        for rank, item in enumerate(exact_results[:top_k], start=1):
            item["rank"] = rank
        return exact_results[:top_k]

    candidate_k = max(candidate_k, top_k)
    keyword_results = keyword_search(vectorstore, query, top_k=max(candidate_k * 2, 12))
    vector_hits = vectorstore.similarity_search_with_score(query, k=max(candidate_k * 2, 12))

    exact_keys = {_result_key(item) for item in exact_results}
    candidates: dict[tuple[str, str, str], dict] = {}
    scores: Counter = Counter()
    rrf_k = 60

    for rank, item in enumerate(keyword_results, start=1):
        key = _result_key(item)
        if key in exact_keys:
            continue
        candidates.setdefault(key, item)
        scores[key] += 1.0 / (rrf_k + rank)

    for rank, (doc, distance) in enumerate(vector_hits, start=1):
        item = _dict_from_hit(0, doc.page_content, doc.metadata, distance, retrieval="vector")
        key = _result_key(item)
        if key in exact_keys:
            continue
        candidates.setdefault(key, item)
        scores[key] += 1.0 / (rrf_k + rank)
        if candidates[key].get("retrieval") == "keyword":
            candidates[key]["retrieval"] = "hybrid"
            candidates[key]["distance"] = min(candidates[key]["distance"], distance)

    fused = sorted(candidates.values(), key=lambda item: scores[_result_key(item)], reverse=True)
    output = _rerank_results(query, [*exact_results, *fused][:candidate_k], top_k)
    for rank, item in enumerate(output, start=1):
        item["rank"] = rank
    return output


def print_search_result(query: str, results: list[dict]) -> None:
    print(f"\n{'=' * 65}")
    print(f"查询：{query}")
    print(f"{'=' * 65}")
    for r in results:
        label = " ".join(part for part in (r.get("law_name"), r.get("article_no"), r.get("topic")) if part)
        print(
            f"\n  [第{r['rank']}名] {label or r['source']} | "
            f"方式：{r.get('retrieval', 'vector')} | 距离/分数：{r['distance']:.4f}"
        )
        print(f"  内容：{r['content'][:120]}...")
    print()


if __name__ == "__main__":
    vectorstore = load_vectorstore()
    count = vectorstore._collection.count()
    print(f"劳动法知识库已加载，共 {count} 个文档块\n")

    if count == 0:
        print("知识库为空，请先准备本地切片并重建向量库：")
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
