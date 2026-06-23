import os
from pathlib import Path

PROJECT_ROOT = os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parent))


def _load_dotenv() -> None:
    """Load local .env values without requiring python-dotenv at runtime."""
    path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.exists(path):
        return
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


_load_dotenv()

# 本地知识库切片缓存
IMPORTED_DATA_PATH = os.getenv(
    "IMPORTED_DATA_PATH",
    os.path.join(PROJECT_ROOT, "data/rag_law/imported/knowledge.json"),
)

# 向量库
DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.join(PROJECT_ROOT, "data/rag_law/chroma_db"))
COLLECTION_NAME = "labor_law_knowledge"

# Embedding 模型（与 day3 实验共用 BGE 小模型）
BGE_MODEL_PATH = os.getenv("BGE_MODEL_PATH", "/root/autodl-tmp/models/bge/AI-ModelScope/bge-small-zh-v1.5")
BGE_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# Reranker 模型：优先使用本地路径，不存在时使用 HuggingFace/镜像下载。
RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", "/root/autodl-tmp/models/bge/BAAI/bge-reranker-base")
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

if not os.path.exists(BGE_MODEL_PATH):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


# 外部工具 / API 配置
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "false").lower() == "true"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
