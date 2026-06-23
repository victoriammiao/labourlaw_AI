#!/usr/bin/env python3
"""一键构建 v3 劳动法 SFT 数据集并注册到 LLaMA-Factory。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from labor_law_v3 import normalize_text, truncate_output

ROOT = Path(__file__).resolve().parent
DAY4_DATA = ROOT.parents[1] / "day4" / "LLaMA-Factory" / "data"
EVAL_QUESTIONS = ROOT.parents[1] / "day4" / "LLaMA-Factory" / "eval" / "disc_law_eval_20.json"
EVAL_COMPARE = (
    ROOT.parents[1]
    / "day4"
    / "LLaMA-Factory"
    / "eval"
    / "results"
    / "compare_v2_20260622_024854.json"
)
DATASET_INFO = DAY4_DATA / "dataset_info.json"

PAIR_OUT = ROOT / "converted" / "disc_law_labor_qa_v3.jsonl"
TRIPLET_OUT = ROOT / "converted" / "disc_law_triplet_labor_qa_v3.jsonl"
GOLD_OUT = ROOT / "converted" / "disc_law_eval_gold_v3.jsonl"
PAIR_RAW = ROOT / "converted" / "disc_law_labor_pair_v3_raw.jsonl"
TRIPLET_RAW = ROOT / "converted" / "disc_law_triplet_labor_v3_raw.jsonl"


def run_py(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / script), *args]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open(encoding="utf-8") if _.strip())


def build_eval_gold(compare_path: Path, out_path: Path, held_out_path: Path) -> int:
    if not compare_path.exists():
        print(f"skip eval gold: missing {compare_path}")
        return 0

    held = {normalize_text(q) for q in json.loads(held_out_path.read_text(encoding="utf-8"))}
    data = json.loads(compare_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for item in data.get("base", []):
        q = str(item.get("question", "")).strip()
        a = truncate_output(str(item.get("answer", "")).strip(), 900)
        if not q or not a or normalize_text(q) not in held:
            continue
        rows.append({"input": q, "output": a, "source": "eval_base_gold"})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"eval gold: {len(rows)} -> {out_path}")
    return len(rows)


def symlink_outputs() -> None:
    DAY4_DATA.mkdir(parents=True, exist_ok=True)
    for src, name in [
        (PAIR_OUT, "disc_law_labor_qa_v3.jsonl"),
        (TRIPLET_OUT, "disc_law_triplet_labor_qa_v3.jsonl"),
        (GOLD_OUT, "disc_law_eval_gold_v3.jsonl"),
    ]:
        dst = DAY4_DATA / name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
        print(f"link {dst} -> {src}")


def register_datasets() -> None:
    info = json.loads(DATASET_INFO.read_text(encoding="utf-8"))
    entries = {
        "disc_law_labor_qa_v3": {
            "file_name": "disc_law_labor_qa_v3.jsonl",
            "columns": {"prompt": "input", "response": "output"},
        },
        "disc_law_triplet_labor_qa_v3": {
            "file_name": "disc_law_triplet_labor_qa_v3.jsonl",
            "columns": {"prompt": "input", "response": "output"},
        },
        "disc_law_eval_gold_v3": {
            "file_name": "disc_law_eval_gold_v3.jsonl",
            "columns": {"prompt": "input", "response": "output"},
        },
    }
    for key, val in entries.items():
        info[key] = val
    DATASET_INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"updated {DATASET_INFO}")


def print_summary() -> None:
    v2_pair = ROOT / "converted" / "disc_law_labor_qa_v2.jsonl"
    v2_triplet = ROOT / "converted" / "disc_law_triplet_labor_qa_v2.jsonl"
    print("\n=== v2 vs v3 条数 ===")
    print(f"  pair:    v2={count_lines(v2_pair):>5}  v3={count_lines(PAIR_OUT):>5}")
    print(f"  triplet: v2={count_lines(v2_triplet):>5}  v3={count_lines(TRIPLET_OUT):>5}")
    print(f"  gold:    v3={count_lines(GOLD_OUT):>5}")
    print(f"  total v3 train (excl identity): {count_lines(PAIR_OUT)+count_lines(TRIPLET_OUT)+count_lines(GOLD_OUT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v3 labor-law SFT datasets.")
    parser.add_argument("--pair-max", type=int, default=3500)
    parser.add_argument("--triplet-max", type=int, default=2000)
    parser.add_argument("--skip-register", action="store_true")
    args = parser.parse_args()
    held = str(EVAL_QUESTIONS.resolve())

    run_py("prepare_labor_subset_v3.py", "--held-out", held)

    run_py(
        "clean_disc_law_v3.py",
        "--input",
        str(PAIR_RAW),
        "--output",
        str(PAIR_OUT),
        "--max-rows",
        str(args.pair_max),
        "--held-out",
        held,
    )
    run_py(
        "clean_disc_law_v3.py",
        "--input",
        str(TRIPLET_RAW),
        "--output",
        str(TRIPLET_OUT),
        "--triplet",
        "--max-rows",
        str(args.triplet_max),
        "--held-out",
        held,
    )

    build_eval_gold(EVAL_COMPARE, GOLD_OUT, EVAL_QUESTIONS)
    symlink_outputs()
    if not args.skip_register:
        register_datasets()
    print_summary()
    print("\n建议 v3 训练 dataset:")
    print("  disc_law_labor_qa_v3,disc_law_triplet_labor_qa_v3,disc_law_eval_gold_v3,identity")


if __name__ == "__main__":
    main()
