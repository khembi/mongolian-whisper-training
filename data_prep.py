#!/usr/bin/env python3
"""Filter raw Mongolian CV audio and compute Whisper mel features."""

from __future__ import annotations

import os

from datasets import Audio, Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import WhisperProcessor

from audio_io import audio_duration_sec, load_audio_array

DEFAULT_DATASET = "Ganaa0614/mongolian-commonvoice-stt-translated-full"
# Train + validation first; test only needed for final eval.
PREP_SPLITS = ("train", "validation", "test")


def dataset_num_proc() -> int | None:
    if os.name == "nt":
        return None
    return min(8, os.cpu_count() or 1)


def prepare_batch(
    batch: dict,
    processor: WhisperProcessor,
    text_column: str,
) -> dict:
    arr, sr = load_audio_array(batch["audio"])
    batch["input_features"] = processor.feature_extractor(
        arr,
        sampling_rate=sr,
        return_tensors="np",
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch[text_column].strip().lower()).input_ids
    return batch


def is_valid_sample(
    batch: dict,
    text_column: str,
    max_duration_sec: float,
) -> bool:
    duration = audio_duration_sec(batch["audio"])
    text = (batch.get(text_column) or "").strip()
    return 0.1 < duration <= max_duration_sec and len(text) > 0


def load_raw_dataset(
    dataset_name: str,
    dataset_config: str,
    max_samples: int | None = None,
) -> DatasetDict:
    raw = load_dataset(dataset_name, dataset_config)
    raw = raw.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    if max_samples:
        raw = DatasetDict(
            {split: ds.select(range(min(max_samples, len(ds)))) for split, ds in raw.items()}
        )
    return raw


def prepare_split(
    raw_split: Dataset,
    processor: WhisperProcessor,
    text_column: str,
    max_duration_sec: float,
    num_proc: int | None,
    desc: str,
) -> Dataset:
    filtered = raw_split.filter(
        lambda batch: is_valid_sample(batch, text_column, max_duration_sec),
        num_proc=num_proc,
        desc=f"Filtering {desc}",
    )

    return filtered.map(
        lambda batch: prepare_batch(batch, processor, text_column),
        remove_columns=filtered.column_names,
        num_proc=num_proc,
        desc=f"Preparing {desc}",
        writer_batch_size=256,
    )


def split_cache_path(cache_dir: str, split: str) -> str:
    return os.path.join(cache_dir, split)


def split_is_cached(cache_dir: str, split: str) -> bool:
    path = split_cache_path(cache_dir, split)
    return os.path.isfile(os.path.join(path, "dataset_info.json"))


def load_or_prepare_splits(
    raw: DatasetDict,
    processor: WhisperProcessor,
    text_column: str,
    max_duration_sec: float,
    cache_dir: str,
    splits: tuple[str, ...] = PREP_SPLITS,
    num_proc: int | None = None,
) -> DatasetDict:
    os.makedirs(cache_dir, exist_ok=True)
    nproc = num_proc if num_proc is not None else dataset_num_proc()
    prepared = DatasetDict()

    for split in splits:
        if split not in raw:
            continue

        cached = split_cache_path(cache_dir, split)
        if split_is_cached(cache_dir, split):
            print(f"Loading cached {split} from {cached}")
            prepared[split] = load_from_disk(cached)
            continue

        print(f"Preparing {split} (will cache to {cached})...")
        prepared[split] = prepare_split(
            raw[split],
            processor,
            text_column,
            max_duration_sec,
            nproc,
            split,
        )
        prepared[split].save_to_disk(cached)
        print(f"Cached {split}: {len(prepared[split])} samples")

    return prepared


def load_prepared_dataset(path: str) -> DatasetDict:
    """Load a DatasetDict saved at root, or per-split dirs from prepare_data.py."""
    if os.path.isfile(os.path.join(path, "dataset_dict.json")):
        return load_from_disk(path)

    prepared = DatasetDict()
    for split in PREP_SPLITS:
        if split_is_cached(path, split):
            prepared[split] = load_from_disk(split_cache_path(path, split))

    if not prepared:
        raise FileNotFoundError(
            f"No prepared splits under {path}. "
            "Run: python prepare_data.py --output-dir ./prepared-dataset"
        )
    return prepared
