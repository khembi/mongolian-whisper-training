#!/usr/bin/env python3
"""
Local smoke test — validates the pipeline before RunPod.

Checks:
  1. Dependencies installed
  2. Dataset downloads and preprocesses
  3. Baseline transcription (whisper-small)
  4. Mini LoRA training (5 steps, whisper-small) — if enough VRAM
  5. Merge LoRA weights

Does NOT train large-v3 locally (needs ~12+ GB free VRAM).

Usage:
  python smoke_test.py
  python smoke_test.py --skip-training
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys

# Use soundfile for audio decoding (avoids torchcodec/FFmpeg on Windows).
os.environ.setdefault("DATASETS_AUDIO_BACKEND", "soundfile")

import torch


def gpu_free_mb() -> int:
    if not torch.cuda.is_available():
        return 0
    free, _ = torch.cuda.mem_get_info()
    return free // (1024 * 1024)


def run_step(name: str, cmd: list[str]) -> bool:
    print(f"\n{'=' * 60}")
    print(f"STEP: {name}")
    print(f"CMD:  {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"\nFAILED: {name} (exit {result.returncode})")
        return False
    print(f"\nPASSED: {name}")
    return True


def test_imports() -> bool:
    print("\n" + "=" * 60)
    print("STEP: Check dependencies")
    print("=" * 60)
    required = ["torch", "transformers", "datasets", "peft", "evaluate", "jiwer", "soundfile"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  OK  {pkg}")
        except ImportError:
            print(f"  MISSING  {pkg}")
            missing.append(pkg)
    if missing:
        print("\nInstall: pip install -r requirements.txt")
        return False
    print("\nPASSED: dependencies")
    return True


def test_dataset_only() -> bool:
    print("\n" + "=" * 60)
    print("STEP: Dataset download + preprocessing")
    print("=" * 60)
    from datasets import Audio, load_dataset
    from transformers import WhisperProcessor

    from audio_io import load_audio_array
    from train import is_valid_sample, prepare_dataset

    ds = load_dataset(
        "Ganaa0614/mongolian-commonvoice-stt-translated-full",
        "default",
        split="train",
    )
    ds = ds.select(range(5)).cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="mn", task="transcribe"
    )

    for i, row in enumerate(ds):
        assert is_valid_sample(row, "sentence", 20.0), f"Sample {i} failed validation"
        prepared = prepare_dataset(row, processor, "sentence", 2000, 448)
        assert "input_features" in prepared and "labels" in prepared
        print(f"  Sample {i}: OK ({len(row['sentence'])} chars)")

    print("\nPASSED: dataset")
    return True


def test_baseline_inference() -> bool:
    print("\n" + "=" * 60)
    print("STEP: Baseline Mongolian inference (whisper-small)")
    print("=" * 60)

    import evaluate as hf_evaluate
    from audio_io import load_audio_array
    from datasets import Audio, load_dataset
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="mn", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()

    ds = load_dataset(
        "Ganaa0614/mongolian-commonvoice-stt-translated-full",
        "default",
        split="test",
    )
    ds = ds.select(range(3)).cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    forced = processor.get_decoder_prompt_ids(language="mn", task="transcribe")
    wer_metric = hf_evaluate.load("wer")
    preds, refs = [], []

    for sample in ds:
        arr, sr = load_audio_array(sample["audio"])
        ref = sample["sentence"].strip().lower()
        inputs = processor(arr, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if device == "cuda":
            inputs["input_features"] = inputs["input_features"].to(torch.float16)
        with torch.no_grad():
            ids = model.generate(**inputs, forced_decoder_ids=forced, max_new_tokens=128)
        pred = processor.batch_decode(ids, skip_special_tokens=True)[0].strip().lower()
        preds.append(pred)
        refs.append(ref)
        print(f"  REF ({len(ref)} chars): OK")
        print(f"  HYP ({len(pred)} chars): OK")
        print()

    wer = 100 * wer_metric.compute(predictions=preds, references=refs)
    print(f"Baseline WER (3 samples): {wer:.1f}%")
    print("(High WER is normal before fine-tuning.)")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nPASSED: baseline inference")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local smoke test before RunPod.")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument(
        "--min-vram-mb",
        type=int,
        default=6000,
        help="Min free VRAM for mini training (default 6000 MB).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    print("Mongolian Whisper — local smoke test")
    print(f"Python: {sys.version.split()[0]}")
    print(f"CUDA:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM free: {gpu_free_mb()} MB")

    steps: list[tuple[str, bool]] = []
    steps.append(("imports", test_imports()))
    if not steps[-1][1]:
        sys.exit(1)
    steps.append(("dataset", test_dataset_only()))
    if not steps[-1][1]:
        sys.exit(1)
    steps.append(("inference", test_baseline_inference()))

    if not args.skip_training:
        free_mb = gpu_free_mb()
        if free_mb < args.min_vram_mb:
            print(
                f"\nSKIPPING mini training: need {args.min_vram_mb} MB free, have {free_mb} MB."
            )
            print("Close GPU apps (Discord, Chrome, Cursor) and re-run.")
        else:
            ok = run_step("mini LoRA training", [
                sys.executable, "train.py",
                "--model-name", "openai/whisper-small",
                "--output-dir", "./output-smoke",
                "--max-samples", "16", "--max-steps", "5",
                "--batch-size", "1", "--grad-accum", "2", "--fp16",
            ])
            steps.append(("mini-train", ok))
            if ok:
                steps.append(("merge", run_step("merge LoRA", [
                    sys.executable, "merge_lora.py",
                    "--base-model", "openai/whisper-small",
                    "--adapter-dir", "./output-smoke/best",
                    "--output-dir", "./merged-smoke",
                    "--device", "cpu",
                ])))

    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    for name, ok in steps:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    if all(ok for _, ok in steps):
        print("\nAll local checks passed. Safe to run on RunPod:")
        print("  bash scripts/run_training.sh")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
