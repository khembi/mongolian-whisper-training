#!/usr/bin/env python3
"""Decode audio samples without torchcodec (Windows-safe)."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import soundfile as sf


def load_audio_array(audio: dict[str, Any], target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """Load mono float32 audio from a datasets Audio column value."""
    if "array" in audio and audio["array"] is not None:
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio.get("sampling_rate") or target_sr)
    elif audio.get("bytes"):
        arr, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        sr = int(sr)
    elif audio.get("path"):
        arr, sr = sf.read(audio["path"], dtype="float32", always_2d=False)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        sr = int(sr)
    else:
        raise ValueError(f"Unsupported audio field: {list(audio.keys())}")

    if sr != target_sr:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)

    return arr.astype(np.float32), target_sr


def audio_duration_sec(audio: dict[str, Any]) -> float:
    if "array" in audio and audio["array"] is not None:
        return len(audio["array"]) / float(audio["sampling_rate"])
    if audio.get("bytes"):
        return sf.info(io.BytesIO(audio["bytes"])).duration
    if audio.get("path"):
        return sf.info(audio["path"]).duration
    return 0.0
