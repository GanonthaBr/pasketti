"""
build_processor.py
Creates a custom Wav2Vec2 processor with our IPA vocabulary.
Run once before training.
"""

import json
from pathlib import Path
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MANIFEST_PATH = Path(r"C:\Users\bgano\Desktop\DataDriven\pasketti\data\drivendata\train_phon_transcripts.jsonl")
VOCAB_SAVE_DIR = Path(r"C:\Users\bgano\Desktop\DataDriven\pasketti\data\ipa_processor")
VOCAB_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 1: Build vocab from data ─────────────────────────────────────────────
print("Building IPA vocabulary from training data...")

with open(MANIFEST_PATH, encoding="utf-8") as f:
    utterances = [json.loads(l) for l in f]

# Collect all unique IPA characters
all_chars = set()
for u in utterances:
    all_chars.update(u["phonetic_text"])

# Build vocab dict — sorted for reproducibility
vocab = {char: idx for idx, char in enumerate(sorted(all_chars))}

# Add special tokens
vocab["[UNK]"] = len(vocab)  # unknown token
vocab["[PAD]"] = len(vocab)  # CTC blank token

print(f"  Vocab size: {len(vocab)} tokens")
print(f"  Tokens: {sorted(vocab.keys())}")

# Save vocab.json (required by Wav2Vec2CTCTokenizer)
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
    word_delimiter_token=" ",
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