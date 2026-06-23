#!/usr/bin/env python3
"""v3 清洗：去复读/列举、Triplet 抽问题、长度与法条密度控制。"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from labor_law_v3 import (
    MULTI_SECTION_RE,
    NUMBERED_LIST_RE,
    extract_law_names,
    extract_triplet_question,
    normalize_text,
    truncate_output,
)

MIN_OUTPUT_CHARS = 40
MAX_OUTPUT_CHARS = 1000
REPEAT_FRAGMENT_LEN = 6
REPEAT_FRAGMENT_MIN_COUNT = 6
MIN_UNIQUE_CHAR_RATIO = 0.22
LAW_HEAVY_THRESHOLD = 4


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
    if not inp:
        return False
    if out.count(inp) >= 2:
        return True
    if len(inp) >= 8 and out.startswith(inp) and len(out) < len(inp) * 4:
        return True
    head = out[: min(80, len(out))]
    if len(inp) >= 10 and inp in head:
        return True
    return False


def is_numbered_list_style(output_text: str) -> bool:
    if NUMBERED_LIST_RE.match(output_text):
        return True
    # 1、... 2、... 3、
    hits = len(re.findall(r"(?:^|\n)\s*\d+[、.)．]", output_text))
    return hits >= 3


def is_multi_section_style(output_text: str) -> bool:
    return len(MULTI_SECTION_RE.findall(output_text)) >= 2


def is_law_heavy(output_text: str) -> bool:
    return len(extract_law_names(output_text)) >= LAW_HEAVY_THRESHOLD


def row_to_pair(row: dict, *, triplet: bool) -> tuple[str, str] | None:
    out = truncate_output(str(row.get("output", "")).strip(), MAX_OUTPUT_CHARS)
    if triplet:
        question = extract_triplet_question(str(row.get("input", "")))
        if not question:
            return None
        return question, out
    return str(row.get("input", "")).strip(), out


def should_drop(inp: str, out: str) -> str | None:
    if not inp or not out:
        return "empty"
    if len(out) < MIN_OUTPUT_CHARS:
        return "too_short"
    if len(out) > MAX_OUTPUT_CHARS + 50:
        return "too_long"
    if is_echo_input(inp, out):
        return "echo"
    if is_repetitive_output(out):
        return "repetitive"
    return None


def clean_file(
    src: Path,
    dst: Path,
    *,
    triplet: bool,
    numbered_downsample: float,
    multi_section_downsample: float,
    law_heavy_downsample: float,
    seed: int,
    max_rows: int,
    held_out: set[str],
) -> dict:
    rng = random.Random(seed)
    stats: dict = {
        "total": 0,
        "kept": 0,
        "dropped": Counter(),
        "downsampled": Counter(),
        "dedup_input": 0,
        "dedup_output": 0,
        "held_out": 0,
        "triplet_no_question": 0,
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

            pair = row_to_pair(row, triplet=triplet)
            if pair is None:
                stats["triplet_no_question"] += 1
                continue
            inp, out = pair

            reason = should_drop(inp, out)
            if reason:
                stats["dropped"][reason] += 1
                continue

            norm_in = normalize_text(inp)
            if norm_in in held_out:
                stats["held_out"] += 1
                continue
            if norm_in in seen_input:
                stats["dedup_input"] += 1
                continue
            if out in seen_output:
                stats["dedup_output"] += 1
                continue

            if is_numbered_list_style(out) and rng.random() > numbered_downsample:
                stats["downsampled"]["numbered"] += 1
                continue
            if is_multi_section_style(out) and rng.random() > multi_section_downsample:
                stats["downsampled"]["multi_section"] += 1
                continue
            if is_law_heavy(out) and rng.random() > law_heavy_downsample:
                stats["downsampled"]["law_heavy"] += 1
                continue

            seen_input.add(norm_in)
            seen_output.add(out)
            kept_rows.append({"input": inp, "output": out})
            if max_rows and len(kept_rows) >= max_rows:
                break

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fout:
        for row in kept_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats["kept"] = len(kept_rows)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean DISC labor-law jsonl for v3 SFT.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--triplet", action="store_true", help="Extract <问题> as input")
    parser.add_argument("--numbered-downsample", type=float, default=0.2)
    parser.add_argument("--multi-section-downsample", type=float, default=0.1)
    parser.add_argument("--law-heavy-downsample", type=float, default=0.3)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--held-out", default="", help="JSON list of eval questions")
    args = parser.parse_args()

    held_out: set[str] = set()
    if args.held_out:
        data = json.loads(Path(args.held_out).read_text(encoding="utf-8"))
        if isinstance(data, list):
            held_out = {normalize_text(q) for q in data if isinstance(q, str)}

    stats = clean_file(
        Path(args.input),
        Path(args.output),
        triplet=args.triplet,
        numbered_downsample=args.numbered_downsample,
        multi_section_downsample=args.multi_section_downsample,
        law_heavy_downsample=args.law_heavy_downsample,
        seed=args.seed,
        max_rows=args.max_rows,
        held_out=held_out,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2, default=str))
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
