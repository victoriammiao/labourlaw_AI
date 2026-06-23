#!/usr/bin/env python3
"""Batch Chat eval: base Qwen2.5-7B vs disc_law LoRA (20 labor-law questions)."""

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
ADAPTER = str(
    FACTORY / "saves/Qwen2.5-7B-Instruct/lora/disc_law_qwen7b_v1"
)
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


def run_one(label: str, model_args: dict, questions: list[str]) -> list[dict]:
    chat = ChatModel({**COMMON_ARGS, **model_args})
    rows: list[dict] = []
    for idx, question in enumerate(questions, start=1):
        messages = [{"role": "user", "content": question}]
        answer = chat.chat(messages)[0].response_text
        rows.append({"id": idx, "question": question, "answer": answer})
        print(f"[{label}] {idx}/{len(questions)} OK", flush=True)
    del chat
    torch_gc()
    return rows


def write_markdown(path: Path, base_rows: list[dict], lora_rows: list[dict]) -> None:
    lines = [
        "# DISC Law LoRA vs Base — 20 题对比",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- 基座：`{BASE_MODEL}`",
        f"- LoRA：`{ADAPTER}`",
        "",
    ]
    for b, l in zip(base_rows, lora_rows):
        lines.extend(
            [
                f"## Q{b['id']}. {b['question']}",
                "",
                "### 基座",
                b["answer"],
                "",
                "### LoRA (disc_law_qwen7b_v1)",
                l["answer"],
                "",
                "---",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=== Evaluating BASE ===", flush=True)
    base_rows = run_one("base", {"model_name_or_path": BASE_MODEL}, questions)

    print("=== Evaluating LoRA ===", flush=True)
    lora_rows = run_one(
        "lora",
        {"model_name_or_path": BASE_MODEL, "adapter_name_or_path": ADAPTER},
        questions,
    )

    payload = {
        "base_model": BASE_MODEL,
        "adapter": ADAPTER,
        "questions": questions,
        "base": base_rows,
        "lora": lora_rows,
    }
    json_path = OUT_DIR / f"compare_{stamp}.json"
    md_path = OUT_DIR / f"compare_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, base_rows, lora_rows)
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
