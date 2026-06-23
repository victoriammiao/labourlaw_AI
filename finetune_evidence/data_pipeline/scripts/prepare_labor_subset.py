#!/usr/bin/env python3
"""从 DISC-Law-SFT Pair-QA jsonl 筛出劳动法相关样本，转为 LLaMA Factory 可直接用的 jsonl。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LABOR_KEYWORDS = (
    "劳动",
    "工伤",
    "劳动合同",
    "辞退",
    "加班",
    "工资",
    "社保",
    "仲裁",
    "劳动争议",
    "派遣",
    "竞业",
    "赔偿",
    "解雇",
    "用工",
)


def is_labor_related(text: str) -> bool:
    return any(k in text for k in LABOR_KEYWORDS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="raw/DISC-Law-SFT-Pair-QA-released.jsonl",
        help="下载后的 DISC Pair-QA jsonl",
    )
    parser.add_argument(
        "--output",
        default="converted/disc_law_labor_qa.jsonl",
        help="输出 jsonl（字段仍为 input/output）",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="0 表示不限制")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            text = f"{row.get('input', '')} {row.get('output', '')}"
            if not is_labor_related(text):
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            if args.max_rows and kept >= args.max_rows:
                break

    print(f"扫描 {total} 条，保留劳动法相关 {kept} 条 -> {dst}")


if __name__ == "__main__":
    main()
