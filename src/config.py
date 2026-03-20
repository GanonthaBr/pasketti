"""
config.py
Central configuration for all paths and hyperparameters.
Automatically detects Windows vs Linux.
"""

import os
import platform
from pathlib import Path

# ── Base directory (always the project root) ──────────────────────────────────
BASE_DIR = Path(__file__).parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR        = BASE_DIR / "data"
DATA_DD         = DATA_DIR / "drivendata"
DATA_TB         = DATA_DIR / "talkbank"
DATA_COMBINED   = DATA_DIR / "combined"
PROCESSOR_DIR   = DATA_DIR / "ipa_processor"
VOCAB_FILE      = DATA_DIR / "vocab.json"

MANIFEST_DD     = DATA_DD / "train_phon_transcripts.jsonl"
MANIFEST_TB     = DATA_TB / "train_phon_transcripts.jsonl"
MANIFEST_COMBINED = DATA_COMBINED / "train.jsonl"
MANIFEST_VAL    = DATA_COMBINED / "val.jsonl"

# ── Model paths ───────────────────────────────────────────────────────────────
MODELS_DIR      = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
BEST_DIR        = MODELS_DIR / "best"

# ── Logs ──────────────────────────────────────────────────────────────────────
LOGS_DIR        = BASE_DIR / "logs"

# ── Audio settings ────────────────────────────────────────────────────────────
TARGET_SR       = 16_000
MIN_DURATION    = 0.3
MAX_DURATION    = 15.0

# ── Training defaults ─────────────────────────────────────────────────────────
# These are overridden by command line args in train.py
DEFAULT_MODEL   = "facebook/wav2vec2-base"
LARGE_MODEL     = "facebook/wav2vec2-large-960h"
WAVLM_MODEL     = "microsoft/wavlm-large"

# ── Create directories if they don't exist ────────────────────────────────────
for d in [
    DATA_DD, DATA_TB, DATA_COMBINED,
    PROCESSOR_DIR, CHECKPOINTS_DIR,
    BEST_DIR, LOGS_DIR
]:
    d.mkdir(parents=True, exist_ok=True)

# ── Print config (useful for debugging on cluster) ────────────────────────────
def print_config():
    print("CONFIG")
    print(f"  Platform:    {platform.system()}")
    print(f"  Base dir:    {BASE_DIR}")
    print(f"  Data DD:     {DATA_DD}")
    print(f"  Data TB:     {DATA_TB}")
    print(f"  Processor:   {PROCESSOR_DIR}")
    print(f"  Checkpoints: {CHECKPOINTS_DIR}")
    print(f"  DD manifest: {MANIFEST_DD} "
          f"({'✅' if MANIFEST_DD.exists() else '❌'})")
    print(f"  TB manifest: {MANIFEST_TB} "
          f"({'✅' if MANIFEST_TB.exists() else '❌'})")

if __name__ == "__main__":
    print_config()