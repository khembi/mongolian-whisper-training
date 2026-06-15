#!/usr/bin/env python3
"""
Merge LoRA adapters into the base Whisper model for whisper.cpp conversion.

Usage:
  python merge_lora.py --adapter-dir ./output/best --output-dir ./merged-model
"""

from __future__ import annotations

import argparse
import os

import evaluate  # noqa: F401 — must load before peft on Windows (Py3.13)
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

DEFAULT_BASE = "openai/whisper-large-v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base Whisper model.")
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--adapter-dir", default="./output/best")
    parser.add_argument("--output-dir", default="./merged-model")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for merge (cpu is safer on Windows after training).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"Loading base model: {args.base_model} ({args.device}, {dtype})")
    base = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if args.device == "cuda" and torch.cuda.is_available():
        base = base.to("cuda")

    print(f"Loading LoRA adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True)

    processor = WhisperProcessor.from_pretrained(args.adapter_dir)
    processor.save_pretrained(args.output_dir)

    size_mb = sum(
        os.path.getsize(os.path.join(args.output_dir, f))
        for f in os.listdir(args.output_dir)
        if f.endswith((".safetensors", ".bin"))
    ) / (1024 * 1024)
    print(f"Done. Merged model size: ~{size_mb:.0f} MB")
    print("Next: run scripts/convert_to_ggml.sh to create whisper.cpp model.")


if __name__ == "__main__":
    main()
