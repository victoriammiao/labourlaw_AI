#!/usr/bin/env python3
"""从 DISC 原始 jsonl 严格筛出劳动法子集（v3）。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from labor_law_v3 import is_strict_labor_row, normalize_text


def subset_file(
    src: Path,
    dst: Path,
    *,
    triplet: bool,
    max_rows: int,
    held_out: set[str],
) -> dict:
    stats: dict = {
        "total": 0,
        "kept": 0,
        "held_out": 0,
        "reasons": Counter(),
        "reject_tags": Counter(),
    }
    seen_input: set[str] = set()
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            row = json.loads(line)

            ok, tag = is_strict_labor_row(row, triplet=triplet)
            if not ok:
                stats["reject_tags"][tag] += 1
                continue

            inp = str(row.get("input", "")).strip()
            norm = normalize_text(inp)
            if norm in held_out:
                stats["held_out"] += 1
                continue
            if norm in seen_input:
                continue

            seen_input.add(norm)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["reasons"][tag] += 1
            stats["kept"] += 1
            if max_rows and stats["kept"] >= max_rows:
                break

    return stats


def load_held_out(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {normalize_text(q) for q in data if isinstance(q, str)}
    return set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict labor-law subset for v3.")
    parser.add_argument(
        "--pair-input",
        default="raw/DISC-Law-SFT-Pair-QA-released.jsonl",
    )
    parser.add_argument(
        "--triplet-input",
        default="raw/DISC-Law-SFT-Triplet-QA-released.jsonl",
    )
    parser.add_argument(
        "--pair-output",
        default="converted/disc_law_labor_pair_v3_raw.jsonl",
    )
    parser.add_argument(
        "--triplet-output",
        default="converted/disc_law_triplet_labor_v3_raw.jsonl",
    )
    parser.add_argument("--pair-max", type=int, default=0, help="0 = no cap")
    parser.add_argument("--triplet-max", type=int, default=0, help="0 = no cap")
    parser.add_argument(
        "--held-out",
        default="../../day4/LLaMA-Factory/eval/disc_law_eval_20.json",
        help="Eval questions to exclude from training",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    held_out = load_held_out((root / args.held_out).resolve())

    pair_stats = subset_file(
        root / args.pair_input,
        root / args.pair_output,
        triplet=False,
        max_rows=args.pair_max,
        held_out=held_out,
    )
    triplet_stats = subset_file(
        root / args.triplet_input,
        root / args.triplet_output,
        triplet=True,
        max_rows=args.triplet_max,
        held_out=held_out,
    )

    print("=== Pair subset ===")
    print(json.dumps(pair_stats, ensure_ascii=False, indent=2, default=str))
    print("=== Triplet subset ===")
    print(json.dumps(triplet_stats, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
