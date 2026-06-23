#!/usr/bin/env python3
"""清洗 DISC 劳动法 jsonl：去重、去低质、去复读、列表体降采样。"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

MIN_OUTPUT_CHARS = 40
MAX_OUTPUT_CHARS = 1600
REPEAT_FRAGMENT_LEN = 6
REPEAT_FRAGMENT_MIN_COUNT = 8
MIN_UNIQUE_CHAR_RATIO = 0.25
NUMBERED_LIST_RE = re.compile(r"^\s*1[、.)．]\s*")
NORMALIZE_RE = re.compile(r"[\s\u3000，。！？、；：""''（）()\[\]【】]+")


def normalize_text(text: str) -> str:
    return NORMALIZE_RE.sub("", text.strip().lower())


def is_repetitive_output(text: str) -> bool:
    if len(text) < 80:
        return False
    for i in range(min(40, len(text) - REPEAT_FRAGMENT_LEN)):
        frag = text[i : i + REPEAT_FRAGMENT_LEN]
        if frag.strip() and text.count(frag) >= REPEAT_FRAGMENT_MIN_COUNT:
            return True
    chars = text.replace(" ", "")
    if chars and len(set(chars)) / len(chars) < MIN_UNIQUE_CHAR_RATIO:
        return True
    return False


def is_echo_input(input_text: str, output_text: str) -> bool:
    inp = input_text.strip()
    out = output_text.strip()
    if not inp or len(out) > 300:
        return False
    if inp in out and len(out) < len(inp) * 3:
        return True
    if out.count(inp) >= 3:
        return True
    return False


def is_numbered_list_style(output_text: str) -> bool:
    return bool(NUMBERED_LIST_RE.match(output_text))


def should_drop(row: dict) -> str | None:
    inp = str(row.get("input", "")).strip()
    out = str(row.get("output", "")).strip()
    if not inp or not out:
        return "empty"
    if len(out) < MIN_OUTPUT_CHARS:
        return "too_short"
    if len(out) > MAX_OUTPUT_CHARS:
        return "too_long"
    if is_echo_input(inp, out):
        return "echo"
    if is_repetitive_output(out):
        return "repetitive"
    return None


def clean_file(
    src: Path,
    dst: Path,
    numbered_downsample: float,
    seed: int,
    max_rows: int,
) -> dict:
    rng = random.Random(seed)
    stats = {
        "total": 0,
        "kept": 0,
        "dropped": {},
        "dedup_input": 0,
        "dedup_output": 0,
        "numbered_skipped": 0,
    }

    seen_input: set[str] = set()
    seen_output: set[str] = set()
    kept_rows: list[dict] = []

    with src.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            row = json.loads(line)

            reason = should_drop(row)
            if reason:
                stats["dropped"][reason] = stats["dropped"].get(reason, 0) + 1
                continue

            inp = str(row["input"]).strip()
            out = str(row["output"]).strip()

            norm_in = normalize_text(inp)
            if norm_in in seen_input:
                stats["dedup_input"] += 1
                continue
            if out in seen_output:
                stats["dedup_output"] += 1
                continue

            if is_numbered_list_style(out) and rng.random() > numbered_downsample:
                stats["numbered_skipped"] += 1
                continue

            seen_input.add(norm_in)
            seen_output.add(out)
            kept_rows.append({"input": inp, "output": out, **{k: v for k, v in row.items() if k not in ("input", "output")}})
            if max_rows and len(kept_rows) >= max_rows:
                break

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fout:
        for row in kept_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats["kept"] = len(kept_rows)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean DISC labor-law jsonl for v2 SFT.")
    parser.add_argument("--input", required=True, help="Source jsonl")
    parser.add_argument("--output", required=True, help="Cleaned output jsonl")
    parser.add_argument(
        "--numbered-downsample",
        type=float,
        default=0.5,
        help="Keep ratio for answers starting with '1、' / '1.' (default 0.5)",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="0 = no cap after cleaning")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = clean_file(
        Path(args.input),
        Path(args.output),
        numbered_downsample=args.numbered_downsample,
        seed=args.seed,
        max_rows=args.max_rows,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
