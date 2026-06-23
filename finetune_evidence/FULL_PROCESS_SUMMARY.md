# 微调版本迭代总览
本文件汇总 v1、v2、v3、v3.1 的训练配置、数据变化、指标和评测材料。
## 版本对比
| 版本 | 训练配置 | 数据组合 | 学习率/轮数 | 保留材料 | 结果结论 |
| --- | --- | --- | --- | --- | --- |
| v1 | `configs/training/disc_law_qwen7b_lora_sft.yaml` | `disc_law_labor_qa` + `disc_law_triplet_labor_qa` + `identity` | 5e-5 / 2 epoch | 训练日志、20题对比 | 初版验证流程，回答体例已有改善 |
| v2 | `configs/training/disc_law_qwen7b_v2_lora_sft.yaml` | `disc_law_triplet_labor_qa_v2` + `disc_law_labor_qa_v2` + `identity` | 3e-5 / 2 epoch | 日志、指标、loss 图、20题对比 | 最终采用版本，`train_loss=0.4370`，`eval_loss=0.9257` |
| v3 | `configs/training/disc_law_qwen7b_v3_lora_sft.yaml` | v3 清洗数据 + eval gold + identity | 1e-4 / 1.5 epoch | 日志、指标、loss 图、20题对比 | 数据更严但学习率偏大，复读问题仍存在 |
| v3.1 | `configs/training/disc_law_qwen7b_v3_1_lora_sft.yaml` | v3 清洗数据 + eval gold + identity | 3e-5 / 2 epoch | 日志、指标、loss 图 | 调整超参后仍未优于 v2，作为消融记录 |

## 指标汇总
| 版本 | train_loss | eval_loss | train_runtime(s) | epoch | 证据文件 |
| --- | ---: | ---: | ---: | ---: | --- |
| v2 | 0.4370 | 0.9257 | 2700.7 | 2.0 | `metrics/v2/all_results.json` |
| v3 | 1.0675 | 0.9990 | 3198.3 | 1.5 | `metrics/v3/all_results.json` |
| v3_1 | 1.1073 | 1.0207 | 3630.9 | 2.0 | `metrics/v3_1/all_results.json` |

> v1 的完整 output 目录未保留在轻量证明目录中，但保留了 `logs/disc_law_qwen7b_v1_train.log`、训练配置和 `eval/results/compare_20260621_235916.md/json`，可证明初版训练与评测流程。

## 评测结果文件
- v1 对比：`eval/results/compare_20260621_235916.md` / `.json`
- v2 对比：`eval/results/compare_v2_20260622_024854.md` / `.json`
- v3 对比：`eval/results/compare_v3_20260622_231008.md` / `.json`
- v3.1：保留训练指标和 loss 曲线，未单独生成 20 题 Markdown 对比文件。
