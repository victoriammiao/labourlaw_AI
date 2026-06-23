"""Launch the project-owned labor law RAG workflow frontend."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WEB_DEMO = PROJECT_ROOT / "web" / "workflow_frontend.py"
PYTHON = os.getenv("PYTHON", sys.executable)
CUDA13_LIB = os.getenv("CUDA13_LIB", "")


def main() -> None:
    """Build and launch the Gradio frontend command with project defaults."""
    parser = argparse.ArgumentParser(description="Run the Qwen-based labor law RAG UI.")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--rag-api-url", default="http://127.0.0.1:8000/ask")
    parser.add_argument("--rag-top-k", type=int, default=5)
    parser.add_argument(
        "--checkpoint-path",
        default="",
        help="Optional Qwen checkpoint path. If omitted, runs in RAG-only mode.",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument(
        "--lora-adapter-path",
        default=os.getenv("LORA_ADAPTER_PATH", ""),
        help="LoRA adapter path for fine-tuned labor law model. Use empty string to disable.",
    )
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    command = [
        PYTHON,
        str(WEB_DEMO),
        "--server-name",
        args.server_name,
        "--server-port",
        str(args.server_port),
        "--rag-api-url",
        args.rag_api_url,
        "--rag-top-k",
        str(args.rag_top_k),
    ]
    if args.share:
        command.append("--share")
    if args.load_in_4bit:
        command.append("--load-in-4bit")
    if args.load_in_8bit:
        command.append("--load-in-8bit")
    if args.checkpoint_path:
        command.extend(["--checkpoint-path", args.checkpoint_path])
    else:
        command.append("--rag-only")
    if args.lora_adapter_path:
        command.extend(["--lora-adapter-path", args.lora_adapter_path])
    else:
        command.extend(["--lora-adapter-path", ""])

    env = os.environ.copy()
    library_path = env.get("LD_LIBRARY_PATH", "")
    if CUDA13_LIB and os.path.exists(CUDA13_LIB) and CUDA13_LIB not in library_path.split(":"):
        env["LD_LIBRARY_PATH"] = f"{CUDA13_LIB}:{library_path}" if library_path else CUDA13_LIB

    raise SystemExit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()
