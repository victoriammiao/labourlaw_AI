"""从 Dify 导入知识库切片，并保存到本地 JSON。"""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = "/root/autodl-tmp/projects/Labor Law Legal Advisor"
sys.path.insert(0, PROJECT_ROOT)

from config import DIFY_API_KEY, DIFY_BASE_URL, DIFY_DATASET_ID, IMPORTED_DATA_PATH
from data.rag_law.dify_client import DifyClient, DifyClientError


def save_imported_data(payload: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def cmd_list_datasets(client: DifyClient) -> None:
    datasets = client.list_all_datasets()
    if not datasets:
        print("未找到任何知识库。请确认 API Key 是否正确。")
        return

    print(f"共 {len(datasets)} 个知识库：\n")
    for ds in datasets:
        print(f"  名称: {ds.get('name')}")
        print(f"  ID:   {ds.get('id')}")
        print(f"  文档数: {ds.get('document_count', '未知')}")
        print()


def cmd_import(client: DifyClient, dataset_id: str, output_path: str) -> None:
    print(f"正在从 Dify 拉取知识库 {dataset_id} ...")
    payload = client.import_knowledge_base(dataset_id)
    save_imported_data(payload, output_path)

    print(f"\n✅ 导入完成")
    print(f"  文档数: {payload['document_count']}")
    print(f"  切片数: {payload['chunk_count']}")
    print(f"  保存路径: {output_path}")

    if payload["chunk_count"] == 0:
        print("\n⚠️  切片数为 0，请检查：")
        print("  1. 知识库中是否已上传文档")
        print("  2. 文档 indexing_status 是否为 completed")
        print("  3. dataset_id 是否正确")


def main() -> None:
    parser = argparse.ArgumentParser(description="从 Dify 导入知识库到本地")
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出账号下所有知识库（用于获取 dataset_id）",
    )
    parser.add_argument(
        "--dataset-id",
        default=DIFY_DATASET_ID,
        help="知识库 ID（也可用环境变量 DIFY_DATASET_ID）",
    )
    parser.add_argument(
        "--output",
        default=IMPORTED_DATA_PATH,
        help=f"本地保存路径（默认 {IMPORTED_DATA_PATH}）",
    )
    args = parser.parse_args()

    try:
        client = DifyClient(DIFY_BASE_URL, DIFY_API_KEY)
    except DifyClientError as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    if args.list:
        cmd_list_datasets(client)
        return

    if not args.dataset_id:
        print("❌ 请提供知识库 ID：")
        print("  export DIFY_DATASET_ID='你的知识库UUID'")
        print("  或先运行：python import_from_dify.py --list")
        sys.exit(1)

    try:
        cmd_import(client, args.dataset_id, args.output)
    except DifyClientError as exc:
        print(f"❌ 导入失败: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
