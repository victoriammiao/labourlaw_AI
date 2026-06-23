"""Model loading and text generation helpers."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


DEFAULT_CKPT_PATH = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "")


def load_model_tokenizer(args):
    """Load the base causal LM and optionally attach the fine-tuned LoRA adapter."""
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("--load-in-4bit and --load-in-8bit cannot be used together.")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    device_map = "cpu" if args.cpu_only else "auto"
    model_kwargs = {
        "torch_dtype": "auto",
        "device_map": device_map,
    }

    if args.load_in_4bit or args.load_in_8bit:
        if args.cpu_only:
            raise ValueError("bitsandbytes quantization requires GPU mode; remove --cpu-only.")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "4-bit/8-bit loading requires bitsandbytes support. "
                "Install it with: pip install bitsandbytes"
            ) from exc

        if args.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, **model_kwargs).eval()
    model.generation_config.max_new_tokens = 1200

    adapter_path = getattr(args, "lora_adapter_path", "") or ""
    if adapter_path:
        if not Path(adapter_path).is_dir():
            print(f"Warning: LoRA adapter not found at {adapter_path}, running base model only.")
        else:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "LoRA loading requires peft. Install it with: pip install peft"
                ) from exc
            print(f"Loading LoRA adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
            model.eval()

    return model, tokenizer


def is_peft_model(model) -> bool:
    """Return whether the loaded model supports PEFT adapter toggling."""
    try:
        from peft import PeftModel
    except ImportError:
        return False
    return isinstance(model, PeftModel)


@contextmanager
def with_lora_enabled(model, enabled: bool):
    """Temporarily disable the adapter so the UI can compare base and LoRA outputs."""
    if not is_peft_model(model) or enabled:
        yield
        return
    with model.disable_adapter():
        yield


def render_chat(tokenizer, messages: list[dict]) -> str:
    """Apply the model-specific chat template before generation."""
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    use_lora: bool = True,
) -> str:
    """Generate a complete response for planner and non-streaming calls."""
    input_text = render_chat(tokenizer, [{"role": "user", "content": prompt}])
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    with with_lora_enabled(model, use_lora):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def stream_text(model, tokenizer, prompt: str, use_lora: bool = True):
    """Yield generated text chunks for the Gradio streaming response."""
    input_text = render_chat(tokenizer, [{"role": "user", "content": prompt}])
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_prompt=True,
        timeout=60.0,
        skip_special_tokens=True,
    )

    def _run_generate():
        with with_lora_enabled(model, use_lora):
            model.generate(**inputs, streamer=streamer)

    thread = Thread(target=_run_generate)
    thread.start()

    for new_text in streamer:
        yield new_text


def release_cuda_cache() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
