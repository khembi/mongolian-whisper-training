#!/usr/bin/env python3
"""Prepare Mongolian CV dataset offline (CPU). Saves each split to disk for resume."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("DATASETS_AUDIO_BACKEND", "soundfile")

from data_prep import (
    DEFAULT_DATASET,
    PREP_SPLITS,
    load_or_prepare_splits,
    load_raw_dataset,
)
from transformers import WhisperProcessor

DEFAULT_MODEL = "openai/whisper-large-v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Whisper training features and cache to disk."
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--text-column", default="sentence")
    parser.add_argument("--language", default="mn")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--max-duration-sec", type=float, default=20.0)
    parser.add_argument("--output-dir", default="./prepared-dataset")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--splits",
        default=",".join(PREP_SPLITS),
        help="Comma-separated splits to prepare (default: train,validation,test).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())

    print("=" * 60)
    print("Prepare Mongolian Whisper dataset")
    print("=" * 60)
    print(f"Dataset:  {args.dataset_name}")
    print(f"Output:   {args.output_dir}")
    print(f"Splits:   {', '.join(splits)}")
    print("=" * 60)

    raw = load_raw_dataset(args.dataset_name, args.dataset_config, args.max_samples)
    processor = WhisperProcessor.from_pretrained(
        args.model_name,
        language=args.language,
        task=args.task,
    )

    prepared = load_or_prepare_splits(
        raw=raw,
        processor=processor,
        text_column=args.text_column,
        max_duration_sec=args.max_duration_sec,
        cache_dir=args.output_dir,
        splits=splits,
    )

    print("\nDone.")
    for split in prepared:
        print(f"  {split}: {len(prepared[split])} samples → {args.output_dir}/{split}")


if __name__ == "__main__":
    main()
