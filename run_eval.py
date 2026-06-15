#!/usr/bin/env python3
"""
Evaluate a merged Whisper model (or LoRA adapter) on the test split.

Usage:
  python run_eval.py --model-dir ./merged-model
  python run_eval.py --base-model openai/whisper-large-v3 --adapter-dir ./output/best
"""

from __future__ import annotations

import argparse

import evaluate
import torch
from audio_io import load_audio_array
from datasets import Audio, load_dataset
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

DEFAULT_DATASET = "Ganaa0614/mongolian-commonvoice-stt-translated-full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Whisper on Mongolian test split.")
    parser.add_argument("--model-dir", default=None, help="Merged model directory.")
    parser.add_argument("--base-model", default="openai/whisper-large-v3")
    parser.add_argument("--adapter-dir", default=None, help="LoRA adapter (if not merged).")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--text-column", default="sentence")
    parser.add_argument("--language", default="mn")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit eval samples.")
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    if args.model_dir:
        processor = WhisperProcessor.from_pretrained(args.model_dir)
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model_dir,
            torch_dtype=torch.float16,
        )
        return model, processor

    processor = WhisperProcessor.from_pretrained(args.adapter_dir)
    base = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    return model, processor


def main() -> None:
    args = parse_args()
    if not args.model_dir and not args.adapter_dir:
        raise SystemExit("Provide --model-dir or --adapter-dir")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model(args)
    model.to(device)
    model.eval()

    ds = load_dataset(args.dataset_name, "default", split="test")
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    wer_metric = evaluate.load("wer")
    predictions: list[str] = []
    references: list[str] = []

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language,
        task=args.task,
    )

    print(f"Evaluating {len(ds)} samples on {device}...")
    for i, sample in enumerate(ds):
        arr, sr = load_audio_array(sample["audio"])
        reference = sample[args.text_column].strip().lower()

        inputs = processor(arr, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if device == "cuda" and inputs["input_features"].dtype != next(model.parameters()).dtype:
            inputs["input_features"] = inputs["input_features"].to(next(model.parameters()).dtype)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=448,
            )

        pred = processor.batch_decode(generated, skip_special_tokens=True)[0].strip().lower()
        predictions.append(pred)
        references.append(reference)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(ds)} done")

    wer = 100 * wer_metric.compute(predictions=predictions, references=references)
    print(f"\nTest WER: {wer:.2f}%")

    print("\nSample predictions:")
    for idx in [0, 1, 2]:
        if idx < len(predictions):
            print(f"  REF: {references[idx][:80]}")
            print(f"  HYP: {predictions[idx][:80]}")
            print()


if __name__ == "__main__":
    main()
