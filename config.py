# 劳动法法律顾问 RAG 项目配置
import os

PROJECT_ROOT = "/root/autodl-tmp/projects/Labor Law Legal Advisor"


def _load_dotenv() -> None:
    """从项目根目录的 .env 文件加载环境变量（不覆盖已 export 的值）。"""
    for filename in (".env", ".env.example"):
        path = os.path.join(PROJECT_ROOT, filename)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        break


_load_dotenv()

# Dify 知识库 API（云版默认地址）
DIFY_BASE_URL = os.getenv("DIFY_BASE_URL", "https://api.dify.ai/v1").rstrip("/")
DIFY_API_KEY = os.getenv("DIFY_API_KEY", "")
DIFY_DATASET_ID = os.getenv("DIFY_DATASET_ID", "")

# 从 Dify 导入后的本地缓存
IMPORTED_DATA_PATH = os.path.join(PROJECT_ROOT, "data/rag_law/imported/knowledge.json")

# 向量库
DB_PATH = os.path.join(PROJECT_ROOT, "data/rag_law/chroma_db")
COLLECTION_NAME = "labor_law_knowledge"

# Embedding 模型（与 day3 实验共用 BGE 小模型）
BGE_MODEL_PATH = "/root/autodl-tmp/models/bge/AI-ModelScope/bge-small-zh-v1.5"
BGE_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

if not os.path.exists(BGE_MODEL_PATH):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
