# Qwen2.5-7B 劳动法 LoRA 微调证明材料

本目录用于证明本项目实际完成了基于 LLaMA-Factory 的 Qwen2.5-7B 劳动法 QLoRA 微调与评测。为避免 GitHub 仓库过大，本目录只保留可审查、可复现实验过程的小体积材料，不提交 5GB+ checkpoint，也不提交 155MB 的 LoRA 权重文件。

## 1. 微调任务概况

| 项目 | 内容 |
| --- | --- |
| 基座模型 | Qwen2.5-7B-Instruct |
| 训练框架 | LLaMA-Factory |
| 微调方式 | QLoRA / LoRA SFT |
| 量化方式 | 4bit bitsandbytes |
| 目标领域 | 劳动法问答 |
| 训练数据 | `disc_law_triplet_labor_qa_v2` + `disc_law_labor_qa_v2` + `identity` |
| 训练轮数 | 2 epoch |
| LoRA 参数 | `r=16`，`alpha=32`，`dropout=0.05`，`target=all` |
| 最终版本 | `disc_law_qwen7b_v2` |

核心训练配置见：

- [`configs/disc_law_qwen7b_v2_lora_sft.yaml`](./configs/disc_law_qwen7b_v2_lora_sft.yaml)
- [`configs/disc_law_lora_v2_chat.yaml`](./configs/disc_law_lora_v2_chat.yaml)

## 2. 训练结果指标

训练完成后的关键指标如下，原始 JSON 文件保存在 [`metrics/`](./metrics/)：

| 指标 | 数值 |
| --- | ---: |
| `epoch` | 2.0 |
| `train_loss` | 0.4370 |
| `eval_loss` | 0.9257 |
| `train_runtime` | 2700.6786 秒 |
| `train_samples_per_second` | 6.395 |
| `train_steps_per_second` | 0.400 |
| `eval_runtime` | 39.5244 秒 |
| `eval_samples_per_second` | 11.512 |

对应文件：

- [`metrics/train_results.json`](./metrics/train_results.json)
- [`metrics/eval_results.json`](./metrics/eval_results.json)
- [`metrics/all_results.json`](./metrics/all_results.json)
- [`metrics/trainer_log.jsonl`](./metrics/trainer_log.jsonl)
- [`metrics/trainer_state.json`](./metrics/trainer_state.json)

训练日志见：

- [`logs/disc_law_qwen7b_v2_train.log`](./logs/disc_law_qwen7b_v2_train.log)

日志末尾显示训练达到 `1080/1080` steps，最终 `epoch=2.0`，并输出 `eval_loss=0.9256919026374817`。

## 3. Loss 曲线

训练 loss 曲线：

![training_loss](./plots/training_loss.png)

验证集 eval loss 曲线：

![training_eval_loss](./plots/training_eval_loss.png)

## 4. Base vs LoRA 评测

为了验证微调效果，本项目额外准备了 20 道劳动法问答，对比基座模型与 v2 LoRA 的回答表现。

评测材料：

- 评测集：[`eval/disc_law_eval_20.json`](./eval/disc_law_eval_20.json)
- 评测脚本：[`eval/run_base_v2_compare.py`](./eval/run_base_v2_compare.py)
- 对比结果 JSON：[`eval/results/compare_v2_20260622_024854.json`](./eval/results/compare_v2_20260622_024854.json)
- 对比结果 Markdown：[`eval/results/compare_v2_20260622_024854.md`](./eval/results/compare_v2_20260622_024854.md)

典型观察：

1. 在「公司可以随意辞退员工吗？」等问题上，基座模型更偏通用原则解释；v2 LoRA 会直接列出《劳动合同法》第三十九条的具体法定情形。
2. 在「劳动合同法第19条主要内容是什么？」等条号题上，v2 LoRA 更倾向输出试用期期限相关规定，法律答复体例更明显。
3. 在 RAG 强约束场景下，基座和 LoRA 输出可能趋同；无 RAG 或条号精确题更能体现微调收益。

## 5. 为什么没有提交权重文件

本次实际产出的 LoRA 权重文件为：

```text
saves/Qwen2.5-7B-Instruct/lora/disc_law_qwen7b_v2/adapter_model.safetensors
```

该文件约 155MB，超过 GitHub 普通仓库单文件 100MB 限制；完整 `saves/Qwen2.5-7B-Instruct` 目录约 5.2GB，包含多个实验版本和 checkpoint，不适合提交到课程项目仓库。

因此，本仓库只提交以下证明材料：

- 训练配置
- 推理配置
- 训练日志
- 训练与验证指标
- loss 曲线
- Base vs LoRA 评测脚本与结果

这些材料已经可以证明微调流程真实执行过，并能复现实验设置和结果分析。

