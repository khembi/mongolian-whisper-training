#!/usr/bin/env python3
"""
Fine-tune openai/whisper-large-v3 on Mongolian Common Voice data with LoRA.

Usage:
  python train.py
  python train.py --output-dir ./output --epochs 3 --batch-size 16
"""

from __future__ import annotations

import argparse
import os

# Use soundfile for audio decoding (avoids torchcodec/FFmpeg on Windows).
os.environ.setdefault("DATASETS_AUDIO_BACKEND", "soundfile")

from dataclasses import dataclass
from typing import Any

import evaluate
import torch
from datasets import Audio, DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

DEFAULT_DATASET = "Ganaa0614/mongolian-commonvoice-stt-translated-full"
DEFAULT_MODEL = "openai/whisper-large-v3"
DEFAULT_LANGUAGE = "mn"
DEFAULT_TASK = "transcribe"


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper large-v3 for Mongolian ASR (LoRA)."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--text-column", default="sentence")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--task", default=DEFAULT_TASK, choices=["transcribe", "translate"])
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--max-duration-sec", type=float, default=20.0)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", help="Use bf16 (recommended on A100).")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 instead of bf16.")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit samples per split (for local smoke tests).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after N training steps (overrides epochs for quick tests).",
    )
    return parser.parse_args()


from audio_io import audio_duration_sec, load_audio_array


def prepare_dataset(
    batch: dict,
    processor: WhisperProcessor,
    text_column: str,
    max_input_length: int,
    max_label_length: int,
) -> dict:
    arr, sr = load_audio_array(batch["audio"])
    batch["input_features"] = processor.feature_extractor(
        arr,
        sampling_rate=sr,
        return_tensors="np",
    ).input_features[0]

    text = batch[text_column].strip().lower()
    batch["labels"] = processor.tokenizer(text).input_ids
    return batch


def is_valid_sample(
    batch: dict,
    text_column: str,
    max_duration_sec: float,
) -> bool:
    duration = audio_duration_sec(batch["audio"])
    text = (batch.get(text_column) or "").strip()
    return 0.1 < duration <= max_duration_sec and len(text) > 0


def dataset_num_proc() -> int | None:
    """Windows multiprocessing in datasets.filter/map is unreliable."""
    if os.name == "nt":
        return None
    return min(8, os.cpu_count() or 1)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    use_bf16 = args.bf16 or (torch.cuda.is_available() and not args.fp16)
    use_fp16 = args.fp16 and not use_bf16

    print("=" * 60)
    print("Mongolian Whisper large-v3 LoRA training")
    print("=" * 60)
    print(f"Model:      {args.model_name}")
    print(f"Dataset:    {args.dataset_name}")
    print(f"Output:     {args.output_dir}")
    print(f"Precision:  {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")
    print("=" * 60)

    raw = load_dataset(args.dataset_name, args.dataset_config)
    # decode=False avoids torchcodec (broken on many Windows setups)
    raw = raw.cast_column("audio", Audio(sampling_rate=16_000, decode=False))

    if args.max_samples:
        raw = DatasetDict(
            {split: ds.select(range(min(args.max_samples, len(ds)))) for split, ds in raw.items()}
        )
        print(f"Limited to {args.max_samples} samples per split (smoke test mode)")

    processor = WhisperProcessor.from_pretrained(
        args.model_name,
        language=args.language,
        task=args.task,
    )

    max_input_length = int(args.max_duration_sec * 16000 / 160)  # 10ms frames
    max_label_length = 448

    def _prepare(batch: dict) -> dict:
        return prepare_dataset(
            batch,
            processor,
            args.text_column,
            max_input_length,
            max_label_length,
        )

    def _is_valid(batch: dict) -> bool:
        return is_valid_sample(batch, args.text_column, args.max_duration_sec)

    dataset = DatasetDict()
    nproc = dataset_num_proc()
    for split in raw:
        filtered = raw[split].filter(
            _is_valid,
            num_proc=nproc,
            desc=f"Filtering {split}",
        )
        dataset[split] = filtered.map(
            _prepare,
            remove_columns=filtered.column_names,
            num_proc=nproc,
            desc=f"Preparing {split}",
        )

    print(f"Train samples: {len(dataset['train'])}")
    print(f"Val samples:   {len(dataset['validation'])}")
    print(f"Test samples:  {len(dataset['test'])}")

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if (use_bf16 or use_fp16) else torch.float32,
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    def make_inputs_require_grad(_module, _input, output):
        output.requires_grad_(True)

    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    smoke = bool(args.max_steps)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=min(args.warmup_steps, 10) if smoke else args.warmup_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps else -1,
        gradient_checkpointing=True,
        fp16=use_fp16,
        bf16=use_bf16,
        eval_strategy="no" if smoke else "steps",
        eval_steps=args.eval_steps,
        save_strategy="no" if smoke else "steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=1 if smoke else args.logging_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to=[] if smoke else ["tensorboard"],
        predict_with_generate=not smoke,
        generation_max_length=max_label_length,
        load_best_model_at_end=not smoke,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0 if os.name == "nt" else 4,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    print("\nStarting training...")
    trainer.train()

    best_dir = os.path.join(args.output_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)
    processor.save_pretrained(best_dir)
    print(f"\nBest LoRA checkpoint saved to: {best_dir}")

    print("\nEvaluating on test split...")
    if args.max_steps:
        print("(skipped in smoke test mode — use full training for WER eval)")
    else:
        test_metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
        print(f"Test WER: {test_metrics.get('test_wer', test_metrics):.2f}%")

        metrics_path = os.path.join(args.output_dir, "test_metrics.txt")
        with open(metrics_path, "w", encoding="utf-8") as f:
            for key, value in sorted(test_metrics.items()):
                f.write(f"{key}: {value}\n")
        print(f"Metrics written to: {metrics_path}")


if __name__ == "__main__":
    main()
