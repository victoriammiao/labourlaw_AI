# 数据构建与样本统计
本文件记录劳动法微调数据从 DISC-Law-SFT 原始问答中筛选、清洗、注册到 LLaMA-Factory 的过程。为控制仓库体积并尊重原始数据集分发方式，本目录只提交脚本、注册信息、统计表和少量样例，不提交完整训练集。
## 数据文件统计
| 数据文件 | 样本数 | 文件大小 | 说明 |
| --- | ---: | ---: | --- |
| `disc_law_labor_qa.jsonl` | 6000 | 9,503,603 bytes | v1 Pair 劳动法子集 |
| `disc_law_triplet_labor_qa.jsonl` | 2500 | 9,254,173 bytes | v1 Triplet 劳动法子集 |
| `disc_law_labor_qa_v2.jsonl` | 4000 | 6,136,430 bytes | v2 Pair 子集，最终主版本之一 |
| `disc_law_triplet_labor_qa_v2.jsonl` | 5000 | 18,358,831 bytes | v2 Triplet 子集，最终主版本之一 |
| `disc_law_labor_qa_v3.jsonl` | 3500 | 4,202,083 bytes | v3 Pair 清洗版 |
| `disc_law_triplet_labor_qa_v3.jsonl` | 2000 | 2,047,872 bytes | v3 Triplet 清洗版 |
| `disc_law_eval_gold_v3.jsonl` | 20 | 31,797 bytes | v3 加入的 20 条人工/基准评测金标样本 |

## 数据处理脚本
- `scripts/prepare_labor_subset.py`：早期劳动法关键词筛选脚本。
- `scripts/clean_disc_law.py`：v2 相关清洗脚本。
- `scripts/prepare_labor_subset_v3.py`、`scripts/clean_disc_law_v3.py`、`scripts/build_disc_law_v3.py`：v3/v3.1 相关清洗与构建脚本。
- `dataset_info_disc_law_excerpt.json`：LLaMA-Factory 中 DISC-Law 相关数据集注册片段。

## 样例文件
`samples/` 中每个 `.jsonl` 只保留前 5 条样例，用于展示字段格式；完整训练数据不随仓库提交。
