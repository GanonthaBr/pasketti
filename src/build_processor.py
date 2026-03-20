"""
build_processor.py
Creates a custom Wav2Vec2 processor with our IPA vocabulary.
Run once before training.
"""

import json
import sys
from pathlib import Path
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

# ── Paths (dynamic - works on Windows and Linux) ──────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import MANIFEST_DD, PROCESSOR_DIR, print_config

print_config()

MANIFEST_PATH  = MANIFEST_DD
VOCAB_SAVE_DIR = PROCESSOR_DIR
VOCAB_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 1: Build vocab from data ─────────────────────────────────────────────
print("Building IPA vocabulary from training data...")

with open(MANIFEST_PATH, encoding="utf-8") as f:
    utterances = [json.loads(l) for l in f]

# Collect all unique IPA characters (excluding space)
all_chars = set()
for u in utterances:
    all_chars.update(u["phonetic_text"])

# Remove space — handled separately as word delimiter
all_chars.discard(" ")

# Build vocab — sorted, index 0 is a real phone (NOT blank)
vocab = {char: idx for idx, char in enumerate(sorted(all_chars))}

# Add special tokens AFTER all phones
vocab["|"]     = len(vocab)   # word delimiter (replaces space)
vocab["[UNK]"] = len(vocab)   # unknown token
vocab["[PAD]"] = len(vocab)   # CTC blank ← must be last, NOT index 0

print(f"  Vocab size:        {len(vocab)} tokens")
print(f"  PAD/blank index:   {vocab['[PAD]']}  ← must not be 0")
print(f"  Index 0 token:     {[k for k,v in vocab.items() if v==0]}")
print(f"  Tokens: {sorted(vocab.keys())}")

# Verify CTC safety
if vocab["[PAD]"] == 0:
    raise ValueError("PAD is at index 0 — CTC will fail! Fix vocab ordering.")

# Save vocab.json
vocab_file = VOCAB_SAVE_DIR / "vocab.json"
with open(vocab_file, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)
print(f"  Saved vocab to {vocab_file}")

# ── Step 2: Create tokenizer ──────────────────────────────────────────────────
print("\nCreating IPA tokenizer...")

# tokenizer = Wav2Vec2CTCTokenizer(
#     str(vocab_file),
#     unk_token="[UNK]",
#     pad_token="[PAD]",
#     word_delimiter_token=" ",  # space separates words in IPA
# )

tokenizer = Wav2Vec2CTCTokenizer(
    str(vocab_file),
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
    bos_token=None,
    eos_token=None,
)

# ── Step 3: Create feature extractor ─────────────────────────────────────────
print("Creating feature extractor...")

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16_000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)

# ── Step 4: Combine into processor ───────────────────────────────────────────
print("Combining into processor...")

processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
)

# ── Step 5: Save processor ────────────────────────────────────────────────────
processor.save_pretrained(str(VOCAB_SAVE_DIR))
print(f"  Processor saved to {VOCAB_SAVE_DIR}")

# ── Step 6: Verify ────────────────────────────────────────────────────────────
print("\nVerifying processor...")

# Test encode/decode roundtrip
test_text = "ʔə ʔæpɫ"
encoded = processor.tokenizer(test_text).input_ids
decoded = processor.tokenizer.decode(encoded)

print(f"  Original:  '{test_text}'")
print(f"  Encoded:   {encoded}")
print(f"  Decoded:   '{decoded}'")
print(f"  Match:     {test_text == decoded}")
print(f"\n✅ Processor ready!")