#!/bin/bash
# 一键运行（无需 conda activate，直接用 llm_course 环境的 Python）
set -e

PROJECT="/root/autodl-tmp/projects/Labor Law Legal Advisor"
PYTHON="/root/miniconda3/envs/llm_course/bin/python"

cd "$PROJECT"

echo "==> 检查配置..."
$PYTHON -c "
from config import DIFY_API_KEY, DIFY_DATASET_ID, DIFY_BASE_URL
assert DIFY_API_KEY and '替换' not in DIFY_API_KEY, '请在 .env 中填写真实的 DIFY_API_KEY'
assert DIFY_DATASET_ID and '替换' not in DIFY_DATASET_ID, '请在 .env 中填写真实的 DIFY_DATASET_ID'
print(f'  API: {DIFY_BASE_URL}')
print(f'  dataset_id: {DIFY_DATASET_ID[:8]}...')
"

echo "==> 从 Dify 导入知识库..."
$PYTHON import_from_dify.py

echo "==> 向量化入库..."
$PYTHON step2_embedding.py

echo "==> 检索测试..."
$PYTHON step3_search.py

echo "✅ 全部完成"
