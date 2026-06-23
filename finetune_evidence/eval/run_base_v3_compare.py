#!/usr/bin/env python3
"""Base vs v3 LoRA 20题对比，并与 v1/v2 历史结果比复读。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FACTORY = ROOT.parent
sys.path.insert(0, str(FACTORY / "src"))

from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc

BASE_MODEL = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
ADAPTER_V3 = str(FACTORY / "saves/Qwen2.5-7B-Instruct/lora/disc_law_qwen7b_v3")
V2_JSON = ROOT / "results/compare_v2_20260622_024854.json"
V1_JSON = ROOT / "results/compare_20260621_235916.json"
QUESTIONS_PATH = ROOT / "disc_law_eval_20.json"
OUT_DIR = ROOT / "results"

COMMON_ARGS = {
    "template": "qwen",
    "infer_backend": "huggingface",
    "trust_remote_code": True,
    "quantization_bit": 4,
    "quantization_method": "bnb",
    "finetuning_type": "lora",
    "do_sample": True,
    "temperature": 0.3,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "max_new_tokens": 384,
}


def is_repetitive(text: str) -> bool:
    if len(text) < 80:
        return False
    for i in range(min(30, len(text) - 6)):
        frag = text[i : i + 6]
        if frag.strip() and text.count(frag) >= 5:
            return True
    chars = text.replace(" ", "")
    if chars and len(set(chars)) / len(chars) < 0.15:
        return True
    return False


def run_one(label: str, model_args: dict, questions: list[str]) -> list[dict]:
    chat = ChatModel({**COMMON_ARGS, **model_args})
    rows: list[dict] = []
    for idx, question in enumerate(questions, start=1):
        messages = [{"role": "user", "content": question}]
        answer = chat.chat(messages)[0].response_text
        rows.append({"id": idx, "question": question, "answer": answer})
        print(f"[{label}] {idx}/20 OK", flush=True)
    del chat
    torch_gc()
    return rows


def load_historical_lora(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("v2") or data.get("lora") or []


def write_md(path: Path, base_rows, v3_rows) -> None:
    lines = [
        "# Base vs v3 LoRA — 20题 (repetition_penalty=1.1)",
        "",
        f"时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 基座：{BASE_MODEL}",
        f"- v3 LoRA：{ADAPTER_V3}",
        "",
    ]
    for b, v in zip(base_rows, v3_rows):
        rep = " **[复读]**" if is_repetitive(v["answer"]) else ""
        lines += [
            f"## Q{b['id']}. {b['question']}{rep}",
            "",
            "### 基座",
            b["answer"],
            "",
            "### v3 LoRA",
            v["answer"],
            "",
            "---",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=== BASE ===", flush=True)
    base_rows = run_one("base", {"model_name_or_path": BASE_MODEL}, questions)

    print("=== v3 LoRA ===", flush=True)
    v3_rows = run_one(
        "v3",
        {"model_name_or_path": BASE_MODEL, "adapter_name_or_path": ADAPTER_V3},
        questions,
    )

    v1_rows = load_historical_lora(V1_JSON)
    v2_rows = load_historical_lora(V2_JSON)
    v1_rep = [r["id"] for r in v1_rows if is_repetitive(r.get("answer", ""))]
    v2_rep = [r["id"] for r in v2_rows if is_repetitive(r.get("answer", ""))]
    v3_rep = [r["id"] for r in v3_rows if is_repetitive(r.get("answer", ""))]
    print(f"\n复读 v1: Q{v1_rep or '无'}")
    print(f"复读 v2: Q{v2_rep or '无'}")
    print(f"复读 v3: Q{v3_rep or '无'}")

    payload = {
        "base_model": BASE_MODEL,
        "adapter_v3": ADAPTER_V3,
        "repetition_penalty": 1.1,
        "v1_repetitive_ids": v1_rep,
        "v2_repetitive_ids": v2_rep,
        "v3_repetitive_ids": v3_rep,
        "base": base_rows,
        "v3": v3_rows,
        "v2_historical": v2_rows,
        "v1_lora_historical": v1_rows,
    }
    json_path = OUT_DIR / f"compare_v3_{stamp}.json"
    md_path = OUT_DIR / f"compare_v3_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(md_path, base_rows, v3_rows)
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
