"""
train.py
Fine-tunes wav2vec2 with CTC on the Pasketti IPA dataset.

Usage:
    # Quick local test (wav2vec2-base, 100 utterances):
    python src/train.py --quick_test

    # Full local run (wav2vec2-base):
    python src/train.py --model_name facebook/wav2vec2-base

    # Full Bridges2 run (wav2vec2-large):
    python src/train.py --model_name facebook/wav2vec2-large-960h
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate

# Import our dataset
import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import (
    PaskettiDataset,
    DataCollatorCTCWithPadding,
    make_splits,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

from config import (
    DATA_DD, PROCESSOR_DIR, CHECKPOINTS_DIR,
    MANIFEST_DD, BEST_DIR, LOGS_DIR,
    DEFAULT_MODEL, LARGE_MODEL,
    print_config,
)


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Metrics ───────────────────────────────────────────────────────────────────
def make_compute_metrics(processor):
    """
    Returns a compute_metrics function for the HuggingFace Trainer.
    Computes Character Error Rate (= Phone Error Rate for IPA).
    """
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_logits   = pred.predictions
        label_ids     = pred.label_ids

        # Replace -100 (padding) with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions greedily
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred_str = processor.tokenizer.batch_decode(pred_ids)

        # Decode references
        label_str = processor.tokenizer.batch_decode(
            label_ids, group_tokens=False
        )

        # Compute CER (= PER for IPA)
        cer = cer_metric.compute(
            predictions=pred_str,
            references=label_str,
        )

        # Log a few examples
        print("\n--- Sample Predictions ---")
        for ref, hyp in zip(label_str[:3], pred_str[:3]):
            print(f"  REF: {ref}")
            print(f"  HYP: {hyp}")
            print()

        return {"cer": cer}

    return compute_metrics


# ── Model setup ───────────────────────────────────────────────────────────────
def load_model(model_name: str, processor: Wav2Vec2Processor):
    """
    Load pretrained wav2vec2 and replace the LM head
    with one sized for our IPA vocabulary.
    """
    vocab_size = len(processor.tokenizer)
    print(f"  Loading {model_name}...")
    print(f"  IPA vocab size: {vocab_size}")

    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=vocab_size,
        ignore_mismatched_sizes=True,  # replaces original LM head
    )

    # Freeze feature encoder — it extracts basic acoustic features
    # and does not need updating for fine-tuning on small data
    model.freeze_feature_encoder()
    print(f"  Feature encoder frozen ✅")

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M")

    return model


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/wav2vec2-base",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save checkpoints (auto-named if not set)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=5000,
        help="Total training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="LR warmup steps",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=False,
        help="Enable audio augmentation",
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        default=False,
        help="Run on 100 utterances to verify pipeline",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("PASKETTI TRAINING")
    print("=" * 60)
    print(f"  Model:         {args.model_name}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Grad accum:    {args.grad_accum}")
    print(f"  Eff. batch:    {args.batch_size * args.grad_accum}")
    print(f"  Steps:         {args.num_steps}")
    print(f"  LR:            {args.learning_rate}")
    print(f"  Augment:       {args.augment}")
    print(f"  Quick test:    {args.quick_test}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device:        {device}")
    if device == "cuda":
        print(f"  GPU:           {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:          {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # ── Load processor ────────────────────────────────────────────────────────
    print("\nLoading IPA processor...")
    processor = Wav2Vec2Processor.from_pretrained(str(PROCESSOR_DIR))
    print(f"  Vocab size: {len(processor.tokenizer)}")

    # ── Create splits ─────────────────────────────────────────────────────────
    print("\nCreating train/val splits...")
    train_utts, val_utts = make_splits(
        str(MANIFEST_DD),
        val_ratio=0.1,
        seed=args.seed,
    )

    # Quick test mode — tiny subset
    if args.quick_test:
        print("\n⚡ QUICK TEST MODE — using 100 train, 20 val")
        train_utts = train_utts[:100]
        val_utts   = val_utts[:20]

    # ── Build datasets ────────────────────────────────────────────────────────
    print("\nBuilding datasets...")

    # Write split manifests to temp files
    import tempfile

    def utts_to_dataset(utts, augment):
        # Write subset to temp jsonl
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl",
            delete=False, encoding="utf-8"
        )
        for u in utts:
            tmp.write(json.dumps(u, ensure_ascii=False) + "\n")
        tmp.close()

        ds = PaskettiDataset(
            manifest_path=tmp.name,
            audio_base=str(DATA_DD),
            processor=processor,
            augment=augment,
        )
        os.unlink(tmp.name)  # cleanup temp file
        return ds

    train_dataset = utts_to_dataset(train_utts, augment=args.augment)
    val_dataset   = utts_to_dataset(val_utts,   augment=False)

    print(f"  Train: {len(train_dataset):,}")
    print(f"  Val:   {len(val_dataset):,}")

    # ── Data collator ─────────────────────────────────────────────────────────
    collator = DataCollatorCTCWithPadding(
        processor=processor,
        padding=True,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading model...")
    model = load_model(args.model_name, processor)

    # ── Output directory ──────────────────────────────────────────────────────
   # ── Output directory ──────────────────────────────────────────────────────
    if args.output_dir is None:
        model_short = args.model_name.split("/")[-1]
        args.output_dir = str(
            CHECKPOINTS_DIR / f"{model_short}_augment{args.augment}"
        )
    print(f"\nCheckpoints → {args.output_dir}")

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,

        # Steps
        max_steps=args.num_steps,
        warmup_steps=args.warmup_steps,

        # Batch
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        # Optimization
        learning_rate=args.learning_rate,
        weight_decay=0.005,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-8,

        # Precision
        fp16=torch.cuda.is_available(),

        # Evaluation
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,

        # Best model
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,

        # Misc
        save_total_limit=2,       # keep only 2 checkpoints
        dataloader_num_workers=0, # 0 for Windows compatibility
        # group_by_length=True,     # speeds up training
        report_to="none",         # change to "wandb" later
        seed=args.seed,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor.feature_extractor,
        data_collator=collator,
        compute_metrics=make_compute_metrics(processor),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5)
        ],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    trainer.train()

    # ── Save best model ───────────────────────────────────────────────────────
    best_dir = BASE_DIR / "models" / "best" / args.output_dir.split(os.sep)[-1]
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    print(f"\nBest model saved to {best_dir}")

    # ── Final eval ────────────────────────────────────────────────────────────
    print("\nFinal evaluation...")
    metrics = trainer.evaluate()
    print(f"\nFinal CER: {metrics['eval_cer']:.4f}")
    print(f"(Lower is better — random baseline ≈ 1.0)")

    # Save metrics
    metrics_path = best_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()