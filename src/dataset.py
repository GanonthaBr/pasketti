"""
dataset.py
Handles all data loading, preprocessing, and augmentation
for the Pasketti phonetic ASR challenge.
"""

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import TARGET_SR, MIN_DURATION, MAX_DURATION
# ── Constants ────────────────────────────────────────────────────────────────


# Replace constants with config values
TARGET_SAMPLE_RATE = TARGET_SR
MIN_DURATION_SEC   = MIN_DURATION
MAX_DURATION_SEC   = MAX_DURATION


# ── Collator ─────────────────────────────────────────────────────────────────
@dataclass
class DataCollatorCTCWithPadding:
    """
    Pads audio inputs and label sequences to the longest
    item in each batch. Used by the HuggingFace Trainer.
    """
    processor: object
    padding: bool = True

    def __call__(
        self, features: List[Dict]
    ) -> Dict[str, torch.Tensor]:

        # Separate inputs and labels
        input_features = [
            {"input_values": f["input_values"]} for f in features
        ]
        label_features = [
            {"input_ids": f["labels"]} for f in features
        ]

        # Pad audio
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Pad labels — replace padding token id with -100
        # so CTC loss ignores them
        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels

        return batch


# ── Dataset ───────────────────────────────────────────────────────────────────
class PaskettiDataset(Dataset):
    """
    Loads audio + IPA transcripts for the Pasketti challenge.

    Args:
        manifest_path : path to train_phon_transcripts.jsonl
        audio_base    : base directory containing audio/ folder
        processor     : HuggingFace Wav2Vec2Processor
        augment       : whether to apply audio augmentations
        max_duration  : filter out clips longer than this
        min_duration  : filter out clips shorter than this
    """

    def __init__(
        self,
        manifest_path: str,
        audio_base: str,
        processor,
        augment: bool = False,
        min_duration: float = MIN_DURATION_SEC,
        max_duration: float = MAX_DURATION_SEC,
    ):
        self.audio_base = Path(audio_base)
        self.processor  = processor
        self.augment    = augment

        # Load and filter manifest
        raw = self._load_manifest(manifest_path)
        self.utterances = self._filter(raw, min_duration, max_duration)

        print(f"  Loaded {len(self.utterances):,} utterances "
              f"(filtered {len(raw)-len(self.utterances):,})")

        # Pre-compute age weights for weighted sampling
        self.age_weights = self._compute_age_weights()

    # ── Loading ───────────────────────────────────────────────────────────────
    def _load_manifest(self, path: str) -> List[Dict]:
        utterances = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                utterances.append(json.loads(line))
        return utterances

    def _filter(
        self,
        utterances: List[Dict],
        min_dur: float,
        max_dur: float,
    ) -> List[Dict]:
        return [
            u for u in utterances
            if min_dur <= u["audio_duration_sec"] <= max_dur
            and len(u["phonetic_text"].strip()) > 0
        ]

    # ── Sampling weights ──────────────────────────────────────────────────────
    def _compute_age_weights(self) -> List[float]:
        """
        Inverse-frequency weights per age bucket.
        Gives underrepresented ages (8-11, 12+) higher
        sampling probability to counteract imbalance.
        """
        age_counts = {}
        for u in self.utterances:
            age = u["age_bucket"]
            age_counts[age] = age_counts.get(age, 0) + 1

        # Inverse frequency
        age_weight = {
            age: 1.0 / count
            for age, count in age_counts.items()
        }

        weights = [
            age_weight[u["age_bucket"]]
            for u in self.utterances
        ]

        print(f"  Age weights: { {k: f'{v:.6f}' for k,v in age_weight.items()} }")
        return weights

    def get_sampler(self) -> WeightedRandomSampler:
        """Returns a WeightedRandomSampler for the DataLoader."""
        return WeightedRandomSampler(
            weights=self.age_weights,
            num_samples=len(self.utterances),
            replacement=True,
        )

    # ── Audio loading ─────────────────────────────────────────────────────────
    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load audio file and resample to 16kHz mono."""
        full_path = self.audio_base / audio_path
        audio, sr = librosa.load(str(full_path), sr=TARGET_SAMPLE_RATE, mono=True)
        return audio  # float32, shape (T,)

    # ── Augmentation ──────────────────────────────────────────────────────────
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to simulate acoustic variability.
        Each augmentation applied independently with some probability.
        """
        # Speed perturbation ±10%
        # Simulates different speaking rates and ages
        if random.random() < 0.5:
            rate = random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)

        # Pitch shift ±2 semitones
        # Children have higher pitch — helps generalization
        if random.random() < 0.3:
            steps = random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(
                audio, sr=TARGET_SAMPLE_RATE, n_steps=steps
            )

        # Add small amount of white noise
        if random.random() < 0.3:
            noise = np.random.randn(len(audio)) * 0.005
            audio = audio + noise

        # Random volume scaling
        if random.random() < 0.5:
            scale = random.uniform(0.7, 1.3)
            audio = audio * scale

        # Clip to valid range
        audio = np.clip(audio, -1.0, 1.0)

        return audio

    # ── Dataset interface ─────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> Dict:
        utt = self.utterances[idx]

        # Load audio
        audio = self._load_audio(utt["audio_path"])

        # Augment if training
        if self.augment:
            audio = self._augment_audio(audio)

        # Process with HuggingFace processor
        # → normalizes audio, extracts input_values
        inputs = self.processor(
            audio,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="np",
        )
        input_values = inputs.input_values[0]  # shape (T,)

        # Tokenize IPA transcript
        # Tokenize IPA transcript
        text = utt["phonetic_text"].replace(" ", "|")
        labels = self.processor.tokenizer(text).input_ids

        return {
            "input_values": input_values,
            "labels":       labels,
            "utterance_id": utt["utterance_id"],
            "age_bucket":   utt["age_bucket"],
        }


# ── Train/Val split ───────────────────────────────────────────────────────────
def make_splits(
    manifest_path: str,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Creates a speaker-stratified train/val split.

    IMPORTANT: No speaker appears in both train and val.
    This prevents data leakage and gives honest val scores.

    Strategy:
        - Group speakers by age bucket
        - Hold out val_ratio of SPEAKERS per age bucket
        - All utterances of held-out speakers go to val
    """
    random.seed(seed)

    # Load all utterances
    utterances = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            utterances.append(json.loads(line))

    # Group speakers by age bucket
    age_to_speakers: Dict[str, List[str]] = {}
    speaker_to_utterances: Dict[str, List[Dict]] = {}

    for u in utterances:
        age    = u["age_bucket"]
        child  = u["child_id"]

        if age not in age_to_speakers:
            age_to_speakers[age] = []
        if child not in age_to_speakers[age]:
            age_to_speakers[age].append(child)

        if child not in speaker_to_utterances:
            speaker_to_utterances[child] = []
        speaker_to_utterances[child].append(u)

    # Hold out speakers per age bucket
    val_speakers = set()
    for age, speakers in age_to_speakers.items():
        random.shuffle(speakers)
        n_val = max(1, int(len(speakers) * val_ratio))
        val_speakers.update(speakers[:n_val])
        print(f"  Age {age:8s}: "
              f"{len(speakers):3d} speakers → "
              f"{n_val} to val")

    # Split utterances
    train_utts = []
    val_utts   = []
    for u in utterances:
        if u["child_id"] in val_speakers:
            val_utts.append(u)
        else:
            train_utts.append(u)

    print(f"\n  Train: {len(train_utts):,} utterances")
    print(f"  Val:   {len(val_utts):,} utterances")
    print(f"  No speaker overlap: ✅")

    return train_utts, val_utts


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from transformers import Wav2Vec2Processor
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from config import MANIFEST_DD, DATA_DD, PROCESSOR_DIR
    import tempfile, os

    print("Testing dataset.py...")
    print("=" * 50)

    MANIFEST   = str(MANIFEST_DD)
    AUDIO_BASE = str(DATA_DD)
    PROC_DIR   = str(PROCESSOR_DIR)

    # Test splits
    print("\nCreating train/val split:")
    train_utts, val_utts = make_splits(MANIFEST)

    # Load processor
    print("\nLoading processor...")
    processor = Wav2Vec2Processor.from_pretrained(PROC_DIR)

    # Test dataset with 50 utterances
    print("\nCreating dataset (first 50 utterances)...")
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl",
        delete=False, encoding="utf-8"
    )
    with open(MANIFEST, encoding="utf-8") as f:
        lines = f.readlines()[:50]
    tmp.writelines(lines)
    tmp.close()

    dataset = PaskettiDataset(
        manifest_path=tmp.name,
        audio_base=AUDIO_BASE,
        processor=processor,
        augment=True,
    )
    os.unlink(tmp.name)

    print(f"\nDataset length: {len(dataset)}")
    print("Loading first item...")
    item = dataset[0]
    print(f"  input_values shape: {item['input_values'].shape}")
    print(f"  labels:             {item['labels']}")
    print(f"  utterance_id:       {item['utterance_id']}")
    print(f"  age_bucket:         {item['age_bucket']}")
    print("\n✅ dataset.py works correctly!")
