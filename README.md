# 劳动法律 AI 顾问（Labor Law Legal Advisor）

大模型微调与优化课程大作业：基于 **RAG + QLoRA 微调 + LangChain Agent** 的劳动法智能咨询系统。

- 报告：见 [`大作业报告.md`](./大作业报告.md)
- 在线演示：Gradio Web UI + 本地 RAG API
- GitHub：`https://github.com/victoriammiao/labourlaw_AI`

## 主要功能

| 功能 | 说明 |
| ---- | ---- |
| 劳动法 RAG 问答 | 混合检索（精确条号 + 关键词 + 向量 + RRF + BGE Reranker），回答附参考资料卡片 |
| 基座 / LoRA 切换 | Gradio 可选「基座模型」或「微调模型 (LoRA v2)」，对比微调效果 |
| 工作流路由 | 本地 Qwen2.5-7B 判断意图，分流至 RAG 或 Agent |
| Agent 工具调用 | DeepSeek + `labor_law_rag` + Tavily 联网，处理实时政策与文书生成 |
| 多会话 & 附件 | 支持 PDF/Word/TXT 上传解析，结合文件内容回答 |
| 离线训练数据管线 | `DISC-Law-SFT/` 提供 DISC 劳动法子集构建与清洗脚本 |

## 环境配置

### 1. 创建 Conda 环境

```bash
conda create -n llm_course python=3.10 -y
conda activate llm_course
cd "/path/to/Labor Law Legal Advisor"
pip install -r requirements.txt
```

主要依赖：`gradio`、`langchain`、`chromadb`、`sentence-transformers`、`fastapi`、`peft`、`bitsandbytes`。

### 2. 准备模型（本地路径，**不入库**）

| 用途 | 路径示例 | 说明 |
| ---- | -------- | ---- |
| 基座模型 | `/path/to/Qwen2.5-7B-Instruct` | HuggingFace 下载 |
| LoRA v2 | `/path/to/disc_law_qwen7b_v2/` | LLaMA Factory 训练产出，含 `adapter_model.safetensors` |
| BGE Embedding | `BAAI/bge-small-zh-v1.5` | 首次运行自动下载或配置 `config.py` 本地路径 |
| BGE Reranker | `BAAI/bge-reranker-base` | 同上 |

LoRA 训练在 `day4/LLaMA-Factory` 中完成，配置见 `examples/disc_law_qwen7b_v2_lora_sft.yaml`。

### 3. 准备知识库

`knowledge.json` 与 `chroma_db/` 因体积较大未提交到 GitHub，需自行准备：

```bash
# 将三部法规切片放入（或按你的导入流程生成）：
# data/rag_law/imported/knowledge.json

# 构建向量库
python step2_embedding.py
```

可用 `python data/rag_law/chunking.py` 查看结构化法条数量（应为 219 条）。

### 4. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 TAVILY_API_KEY、DEEPSEEK_API_KEY
```

## 如何运行

**先启动 RAG API，再启动前端。**

### 终端 1：RAG API（端口 8000）

```bash
bash run_api.sh
# 验证：curl http://127.0.0.1:8000/health
```

### 终端 2：Gradio 前端（端口 7861）

```bash
python run_qwen_frontend.py \
  --checkpoint-path /path/to/Qwen2.5-7B-Instruct \
  --load-in-4bit \
  --lora-adapter-path /path/to/disc_law_qwen7b_v2 \
  --server-port 7861
```

浏览器访问：`http://<服务器IP>:7861`

### RAG-only 模式（不加载大模型，仅测检索）

```bash
python run_qwen_frontend.py --server-port 7861
```

### 重建向量库后

修改 `knowledge.json` 或切片逻辑后，需重新执行 `python step2_embedding.py` 并**重启** `run_api.sh`。

## 项目结构

```text
Labor Law Legal Advisor/
├── README.md                 # 本文件
├── 大作业报告.md              # 课程报告
├── requirements.txt
├── config.py                 # 路径与 API 配置
├── rag_api.py                # RAG FastAPI 服务
├── step2_embedding.py        # 向量库构建
├── step3_search.py           # 混合检索逻辑
├── run_api.sh / run_qwen_frontend.py
├── data/rag_law/
│   ├── chunking.py           # 法条级切片与清洗
│   └── imported/             # knowledge.json（本地，gitignore）
├── web/                      # Gradio 前端与工作流
│   ├── ui.py
│   ├── workflow.py
│   ├── model_runtime.py      # 基座 + LoRA 加载与切换
│   └── legal_agent.py        # LangChain Agent
├── finetune_evidence/        # 微调全流程证明材料（日志、配置、曲线、评测）
│   ├── README.md
│   ├── FULL_PROCESS_SUMMARY.md
│   ├── configs/              # v1/v2/v3/v3.1 训练配置与推理配置
│   ├── logs/                 # 各版本训练日志
│   ├── metrics/              # train/eval results、trainer_state、trainer_log
│   ├── plots/                # loss 与 eval loss 曲线
│   ├── eval/                 # Base vs LoRA 20题评测脚本和结果
│   └── data_pipeline/        # 数据处理脚本、注册片段、统计和样例
└── DISC-Law-SFT/             # 微调数据构建脚本与 v2 子集
    ├── README.md
    ├── prepare_labor_subset.py
    └── converted/              # 训练用 jsonl（大文件见 .gitignore）
```

## 关键脚本说明

| 文件 | 作用 |
| ---- | ---- |
| `data/rag_law/chunking.py` | 将 `knowledge.json` 清洗并合并为 219 条法条级 Document |
| `step2_embedding.py` | BGE 向量化并写入 ChromaDB |
| `step3_search.py` | 精确条号 → 关键词 → 向量 → RRF → Reranker |
| `web/workflow.py` | 意图路由、RAG 调用、Agent 分流 |
| `web/model_runtime.py` | Qwen 4bit 加载、PEFT LoRA 挂载、`disable_adapter()` 切换 |

## 微调证明材料

微调全流程证明材料放在 [`finetune_evidence/`](./finetune_evidence/) 中，用于证明本项目不是只接入现成模型，而是实际完成了 DISC-Law-SFT 劳动法数据筛选、LLaMA-Factory QLoRA 训练、多版本迭代和 Base vs LoRA 评测。

建议从这两个文件开始查看：

- [`finetune_evidence/README.md`](./finetune_evidence/README.md)：证明材料总入口，说明目录结构、最终采用版本和为什么不提交权重。
- [`finetune_evidence/FULL_PROCESS_SUMMARY.md`](./finetune_evidence/FULL_PROCESS_SUMMARY.md)：v1 → v2 → v3 → v3.1 的训练配置、数据变化、指标与结论总览。

`finetune_evidence/` 结构说明：

| 目录 | 内容 |
| ---- | ---- |
| `configs/training/` | v1、v2、v3、v3.1 的 LLaMA-Factory 训练配置 |
| `configs/inference/` | 基座、LoRA v1/v2/v3 的推理配置 |
| `logs/` | 各版本训练日志，包含训练 step、耗时、loss 等过程记录 |
| `metrics/` | `train_results.json`、`eval_results.json`、`trainer_state.json`、`trainer_log.jsonl` 等训练指标 |
| `plots/` | 各版本 `training_loss.png` 与 `training_eval_loss.png` |
| `eval/` | 20题 Base vs LoRA 对比评测脚本和结果 |
| `data_pipeline/` | DISC-Law-SFT 劳动法子集构建脚本、数据注册片段、样本统计和少量样例 |

没有提交完整模型权重和 checkpoint：v2 的 `adapter_model.safetensors` 约 155MB，超过 GitHub 普通仓库单文件限制；完整 `saves/Qwen2.5-7B-Instruct` 约 5GB，也不适合作业仓库。当前提交的配置、日志、loss 曲线、指标、评测结果和数据处理脚本已经可以证明完整微调流程。

## 提交说明（除模型外）

以下内容**应提交**到 GitHub：全部 `.py` 源码、`web/`、`DISC-Law-SFT/` 脚本与小规模 jsonl、`.env.example`、`requirements.txt`、报告、`run_*.sh`。

以下内容**不要提交**：`.env`、基座/LoRA 权重、`chroma_db/`、`knowledge.json`、DISC 原始大文件（`raw/` 下 90MB+ jsonl）。

## License

课程作业项目，数据集 DISC-Law-SFT 请遵循原数据集许可。
