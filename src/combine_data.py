"""
combine_data.py
Merges DrivenData and TalkBank corpora into a single
train/val split. Run once after TalkBank download completes.

Usage:
    python src/combine_data.py
"""

import json
import random
from pathlib import Path
from collections import Counter

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MANIFEST_DD, MANIFEST_TB,
    MANIFEST_COMBINED, MANIFEST_VAL,
    DATA_COMBINED, print_config,
)
from dataset import make_splits


def main():
    print_config()
    print("\n" + "=" * 50)
    print("COMBINING CORPORA")
    print("=" * 50)

    # ── Load DrivenData ───────────────────────────────────────────────────────
    dd_utts = []
    if MANIFEST_DD.exists():
        with open(MANIFEST_DD, encoding="utf-8") as f:
            dd_utts = [json.loads(l) for l in f]
        print(f"  DrivenData:  {len(dd_utts):,} utterances")
    else:
        print(f"  DrivenData:  ❌ not found")

    # ── Load TalkBank ─────────────────────────────────────────────────────────
    tb_utts = []
    if MANIFEST_TB.exists():
        with open(MANIFEST_TB, encoding="utf-8") as f:
            tb_utts = [json.loads(l) for l in f]
        print(f"  TalkBank:    {len(tb_utts):,} utterances")
    else:
        print(f"  TalkBank:    ❌ not found (run again when downloaded)")

    if not dd_utts and not tb_utts:
        print("No data found!")
        return

    # ── Combine ───────────────────────────────────────────────────────────────
    all_utts = dd_utts + tb_utts
    print(f"\n  Total:       {len(all_utts):,} utterances")

    # Show combined age distribution
    ages = Counter(u["age_bucket"] for u in all_utts)
    print("\n  Combined age distribution:")
    for age, count in sorted(ages.items()):
        pct = count / len(all_utts) * 100
        bar = "█" * int(pct / 2)
        print(f"    {age:8s}: {count:6,} ({pct:.1f}%) {bar}")

    # ── Create combined vocab ─────────────────────────────────────────────────
    all_chars = set()
    for u in all_utts:
        all_chars.update(u["phonetic_text"])
    print(f"\n  Combined vocab: {len(all_chars)} unique phones")

    # ── Speaker-stratified split ──────────────────────────────────────────────
    print("\n  Creating speaker-stratified split...")

    # Tag each utterance with its source
    for u in dd_utts:
        u["source"] = "drivendata"
    for u in tb_utts:
        u["source"] = "talkbank"

    # Split per corpus to avoid speaker leakage across sources
    train_utts, val_utts = [], []

    if dd_utts:
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl",
            delete=False, encoding="utf-8"
        )
        for u in dd_utts:
            tmp.write(json.dumps(u, ensure_ascii=False) + "\n")
        tmp.close()
        dd_train, dd_val = make_splits(tmp.name, val_ratio=0.1)
        os.unlink(tmp.name)
        train_utts.extend(dd_train)
        val_utts.extend(dd_val)

    if tb_utts:
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl",
            delete=False, encoding="utf-8"
        )
        for u in tb_utts:
            tmp.write(json.dumps(u, ensure_ascii=False) + "\n")
        tmp.close()
        tb_train, tb_val = make_splits(tmp.name, val_ratio=0.1)
        os.unlink(tmp.name)
        train_utts.extend(tb_train)
        val_utts.extend(tb_val)

    # Shuffle
    random.seed(42)
    random.shuffle(train_utts)
    random.shuffle(val_utts)

    print(f"\n  Final train: {len(train_utts):,}")
    print(f"  Final val:   {len(val_utts):,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    DATA_COMBINED.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_COMBINED, "w", encoding="utf-8") as f:
        for u in train_utts:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")

    with open(MANIFEST_VAL, "w", encoding="utf-8") as f:
        for u in val_utts:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")

    print(f"\n  Saved train → {MANIFEST_COMBINED}")
    print(f"  Saved val   → {MANIFEST_VAL}")
    print(f"\n✅ Done! Run build_processor.py next to rebuild vocab.")


if __name__ == "__main__":
    main()