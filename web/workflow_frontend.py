"""Entry point for the project-owned labor law RAG workflow frontend."""

from __future__ import annotations

from argparse import ArgumentParser
import sys


PROJECT_ROOT = "/root/autodl-tmp/projects/Labor Law Legal Advisor"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from web.model_runtime import DEFAULT_CKPT_PATH, DEFAULT_LORA_ADAPTER_PATH, load_model_tokenizer
from web.ui import launch_demo


def _get_args():
    parser = ArgumentParser(description="Labor law RAG workflow frontend.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CKPT_PATH,
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")
    parser.add_argument("--share", action="store_true", default=False)
    parser.add_argument("--inbrowser", action="store_true", default=False)
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument(
        "--rag-api-url",
        type=str,
        default="http://127.0.0.1:8000/ask",
        help="Labor law RAG API endpoint.",
    )
    parser.add_argument("--rag-top-k", type=int, default=5)
    parser.add_argument("--disable-rag", action="store_true")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the model with bitsandbytes 4-bit quantization.",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load the model with bitsandbytes 8-bit quantization.",
    )
    parser.add_argument(
        "--rag-only",
        action="store_true",
        help="Show retrieved evidence without loading Qwen weights.",
    )
    parser.add_argument(
        "--lora-adapter-path",
        type=str,
        default=DEFAULT_LORA_ADAPTER_PATH,
        help="Optional LoRA adapter path. Leave empty to run base model only.",
    )
    return parser.parse_args()


def main():
    args = _get_args()
    if args.rag_only:
        model, tokenizer = None, None
    else:
        if not args.lora_adapter_path:
            args.lora_adapter_path = ""
        model, tokenizer = load_model_tokenizer(args)
    launch_demo(args, model, tokenizer)


if __name__ == "__main__":
    main()
