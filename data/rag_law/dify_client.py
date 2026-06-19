"""Dify 知识库 API 客户端：拉取文档与切片到本地。"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class DifyClientError(Exception):
    pass


class DifyClient:
    def __init__(self, base_url: str, api_key: str):
        if not api_key:
            raise DifyClientError(
                "未设置 DIFY_API_KEY。请在终端执行：\n"
                "  export DIFY_API_KEY='dataset-xxx'\n"
                "或在 Dify 控制台 → 知识库 → API 中创建密钥。"
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _request(self, method: str, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        if params:
            query = urllib.parse.urlencode(params, doseq=True)
            url = f"{url}?{query}"

        req = urllib.request.Request(
            url,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "LaborLawRAG/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise DifyClientError(f"HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise DifyClientError(f"网络请求失败: {exc.reason}") from exc

    def list_datasets(self, page: int = 1, limit: int = 20) -> dict:
        return self._request("GET", "/datasets", {"page": page, "limit": limit})

    def list_all_datasets(self) -> list[dict]:
        datasets: list[dict] = []
        page = 1
        while True:
            data = self.list_datasets(page=page, limit=100)
            datasets.extend(data.get("data", []))
            if not data.get("has_more"):
                break
            page += 1
        return datasets

    def list_documents(self, dataset_id: str, page: int = 1, limit: int = 100) -> dict:
        return self._request(
            "GET",
            f"/datasets/{dataset_id}/documents",
            {"page": page, "limit": limit},
        )

    def list_all_documents(self, dataset_id: str) -> list[dict]:
        documents: list[dict] = []
        page = 1
        while True:
            data = self.list_documents(dataset_id, page=page, limit=100)
            documents.extend(data.get("data", []))
            if not data.get("has_more"):
                break
            page += 1
        return documents

    def list_segments(
        self,
        dataset_id: str,
        document_id: str,
        page: int = 1,
        limit: int = 100,
    ) -> dict:
        return self._request(
            "GET",
            f"/datasets/{dataset_id}/documents/{document_id}/segments",
            {"page": page, "limit": limit, "status": ["completed"]},
        )

    def list_all_segments(self, dataset_id: str, document_id: str) -> list[dict]:
        segments: list[dict] = []
        page = 1
        while True:
            data = self.list_segments(dataset_id, document_id, page=page, limit=100)
            segments.extend(data.get("data", []))
            if not data.get("has_more"):
                break
            page += 1
        return segments

    def import_knowledge_base(self, dataset_id: str) -> dict[str, Any]:
        """拉取整个知识库：所有文档 + 已完成索引的切片。"""
        documents = self.list_all_documents(dataset_id)
        chunks: list[dict] = []

        for doc in documents:
            doc_id = doc["id"]
            doc_name = doc.get("name", doc_id)
            status = doc.get("indexing_status", "unknown")
            if status != "completed":
                print(f"  跳过未完成索引的文档：{doc_name}（status={status}）")
                continue

            segments = self.list_all_segments(dataset_id, doc_id)
            for seg in segments:
                if not seg.get("enabled", True):
                    continue
                chunks.append(
                    {
                        "content": seg.get("content", "").strip(),
                        "metadata": {
                            "source": doc_name,
                            "document_id": doc_id,
                            "segment_id": seg.get("id"),
                            "position": seg.get("position"),
                            "word_count": seg.get("word_count"),
                        },
                    }
                )

        return {
            "dataset_id": dataset_id,
            "imported_at": int(time.time()),
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "documents": [
                {
                    "id": d["id"],
                    "name": d.get("name"),
                    "indexing_status": d.get("indexing_status"),
                    "word_count": d.get("word_count"),
                }
                for d in documents
            ],
            "chunks": chunks,
        }
