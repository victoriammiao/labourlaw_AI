# DISC-Law-SFT 接入说明（配合 day4 / LLaMA Factory）
启动web-UI
```bash
cd /root/autodl-tmp/projects/day4/LLaMA-Factory
llamafactory-cli webui

source /root/miniconda3/etc/profile.d/conda.sh
conda activate llamafactory

export GRADIO_SERVER_PORT=7860
export GRADIO_ROOT_PATH=/${JUPYTER_NAME}/proxy/7860

llamafactory-cli webui
```



## 结论：用 day4 训练，本目录只放数据和脚本

| 做什么 | 放哪里 |
|--------|--------|
| **LLaMA Factory / WebUI / 已有 LoRA 经验** | 继续用 `~/autodl-tmp/projects/day4/LLaMA-Factory` |
| **DISC 原始 jsonl、转换脚本、劳动法子集** | 本目录 `DISC-Law-SFT/` |
| **不要** | 把整个 day4 再拷贝一份到本项目（重复、占空间、难维护） |

「模型怎么导入」在训练里通常指两件事：

1. **基座模型（Base Model）**：WebUI 里 **Model path** 填本地路径，例如  
   `/root/autodl-tmp/models/Qwen2.5-7B-Instruct`
2. **训练数据集（Dataset）**：把 jsonl 放进 LLaMA Factory 的 `data/`，并在 `dataset_info.json` 注册

`MsDataset.load(...)` 适合**预览/下载**，LLaMA Factory 训练时读的是 **`data/` + `dataset_info.json`**。

---

## 第一步：下载 DISC-Law-SFT（推荐先 QA 子集）

数据集在 HuggingFace：`ShengbinYue/DISC-Law-SFT`（ModelScope 镜像名可能为 `AI-ModelScope/DISC-Law-SFT`）。

建议先下**已开源的问答子集**（体积较小、和劳动/借贷咨询最相关）：

```bash
cd "/root/autodl-tmp/projects/Labor Law Legal Advisor/DISC-Law-SFT"
mkdir -p raw

# 国内可用 hf-mirror（示例：Pair 问答约 93K 条）
curl -L -o raw/DISC-Law-SFT-Pair-QA-released.jsonl \
  "https://hf-mirror.com/datasets/ShengbinYue/DISC-Law-SFT/resolve/main/DISC-Law-SFT-Pair-QA-released.jsonl"

# 可选：Triplet 问答（input 里已带法条参考，更像 RAG+回答）
curl -L -o raw/DISC-Law-SFT-Triplet-QA-released.jsonl \
  "https://hf-mirror.com/datasets/ShengbinYue/DISC-Law-SFT/resolve/main/DISC-Law-SFT-Triplet-QA-released.jsonl"
```

**Pair 单条格式**（只有 input / output）：

```json
{"id": "...", "input": "违章停车与违法停车是否有区别？", "output": "……"}
```

**Triplet 单条格式**（带 reference 法条 + 拼好的 input）：

```json
{"id": "...", "reference": ["《民法典》…"], "input": "法条…\n<问题>：\n…", "output": "…"}
```

---

## 第二步：链接或复制到 day4 的 data 目录

```bash
DAY4_DATA="/root/autodl-tmp/projects/day4/LLaMA-Factory/data"

ln -sf "/root/autodl-tmp/projects/Labor Law Legal Advisor/DISC-Law-SFT/raw/DISC-Law-SFT-Pair-QA-released.jsonl" \
  "$DAY4_DATA/disc_law_pair_qa.jsonl"

# 可选
ln -sf "/root/autodl-tmp/projects/Labor Law Legal Advisor/DISC-Law-SFT/raw/DISC-Law-SFT-Triplet-QA-released.jsonl" \
  "$DAY4_DATA/disc_law_triplet_qa.jsonl"
```

---

## 第三步：在 dataset_info.json 注册

编辑 `day4/LLaMA-Factory/data/dataset_info.json`，增加：

```json
"disc_law_pair_qa": {
  "file_name": "disc_law_pair_qa.jsonl",
  "columns": {
    "prompt": "input",
    "response": "output"
  }
},
"disc_law_triplet_qa": {
  "file_name": "disc_law_triplet_qa.jsonl",
  "columns": {
    "prompt": "input",
    "response": "output"
  }
}
```

说明：DISC 的 `input` 当作 Alpaca 的「问题」；没有单独 `instruction` 字段时，用 `prompt → input` 即可。

---

## 第四步：LLaMA Factory WebUI 训练（与 day4 实验相同）

```bash
cd /root/autodl-tmp/projects/day4/LLaMA-Factory
conda activate llamafactory   # 必须用此环境，不要用 llm_course
llamafactory-cli webui
```

若报 `No module named 'llamafactory'`，说明 editable 安装路径失效，在本目录重装：

```bash
cd /root/autodl-tmp/projects/day4/LLaMA-Factory
/root/miniconda3/envs/llamafactory/bin/pip install -e .
```

| 配置项 | 建议值 |
|--------|--------|
| **Model path** | `/root/autodl-tmp/models/Qwen2.5-7B-Instruct`（与前端一致） |
| **Template** | `qwen` |
| **Finetuning** | `lora`（单卡 AutoDL 不要 full） |
| **Dataset** | 先只选 `disc_law_pair_qa` |
| **Max samples** | 先试 `2000`～`10000`（全量 9 万+ 很慢） |
| **Cutoff len** | `2048` 或 `4096` |
| **Output dir** | `saves/Qwen2.5-7B-Instruct/lora/disc_law_sft` |

训练完成后：**Export / Merge** 把 LoRA 合并到基座（或 Agent 路径直接加载 LoRA adapter，看部署方式）。

---

## 第五步：和劳动法律 Advisor 项目对接

- **Qwen 前端（路由 + RAG 回答）**：可继续用原模型 + RAG，不必替换。
- **DeepSeek Agent（文书 / 联网）**：合并后的模型用 vLLM 起 OpenAI 接口，改 `.env` 的 `DEEPSEEK_BASE_URL`（注意 RAG API 占 8000，vLLM 用 8001）。

---

## 常见问题

**Q：`MsDataset.load('AI-ModelScope/DISC-Law-SFT')` 能直接训练吗？**  
A：一般不能一步到位。要先落到 jsonl/json，再注册 `dataset_info.json`。

**Q：16 万 / 40 万条都要训吗？**  
A：不必。大作业先用 **Pair-QA 子集 + max_samples** 验证流程，再按需加大。

**Q：和 day4 的 my_sft_data 能混训吗？**  
A：可以。WebUI 里 Dataset 多选 `disc_law_pair_qa,identity` 等，注意比例。

**Q：只想练劳动法？**  
A：用 `prepare_labor_subset.py`（见本目录）从 Pair-QA 里按关键词筛「劳动、工伤、合同、仲裁」等样本，再注册为 `disc_law_labor_qa.jsonl`。
