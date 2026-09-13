"""Dataset indexing, speaker-independent splitting, and feature caching for RAVDESS.

Enforces strict speaker-independent splitting:
- Train: Actors 01 to 18 (9 male, 9 female — 1,080 samples)
- Validation: Actors 19 and 20 (1 male, 1 female — 120 samples)
- Test: Actors 21 to 24 (2 male, 2 female — 240 samples)

Guarantees ZERO speaker leakage between splits.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.audio.features import extract_handcrafted_features, extract_mel_spectrogram
from src.config import (
    DATA_DIR,
    EMOTION_CLASSES,
    PROJECT_ROOT,
    RAVDESS_CODE_TO_EMOTION,
    TEST_SPEAKERS,
    TRAIN_SPEAKERS,
    VAL_SPEAKERS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = PROJECT_ROOT / "data" / "cache"


def index_ravdess_dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Scan RAVDESS directory, deduplicate files, and extract speaker-independent splits.

    Filename format:
    03-01-06-01-02-01-12.wav
    Parts:
    [0] Modality (03 = audio-only)
    [1] Vocal channel (01 = speech)
    [2] Emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)
    [3] Emotional intensity (01=normal, 02=strong)
    [4] Statement (01="Kids...", 02="Dogs...")
    [5] Repetition (01=1st, 02=2nd)
    [6] Actor (01 to 24; odd=male, even=female)

    Returns:
        Pandas DataFrame containing indexed metadata with zero duplicate files.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"RAVDESS dataset directory not found at {data_dir}")

    records = []
    seen_basenames = set()

    for root, _, files in os.walk(data_dir):
        for fname in sorted(files):
            if not fname.endswith(".wav") or not fname.startswith("03-01-"):
                continue

            if fname in seen_basenames:
                continue
            seen_basenames.add(fname)

            parts = fname.replace(".wav", "").split("-")
            if len(parts) != 7:
                continue

            emotion_code = parts[2]
            actor_id = int(parts[6])

            if emotion_code not in RAVDESS_CODE_TO_EMOTION:
                continue

            emotion_label = RAVDESS_CODE_TO_EMOTION[emotion_code]
            gender = "male" if (actor_id % 2 != 0) else "female"

            # Assign strict speaker-independent split
            if actor_id in TRAIN_SPEAKERS:
                split = "train"
            elif actor_id in VAL_SPEAKERS:
                split = "val"
            elif actor_id in TEST_SPEAKERS:
                split = "test"
            else:
                split = "ignored"

            full_path = Path(root) / fname
            records.append({
                "path": str(full_path),
                "filename": fname,
                "actor_id": actor_id,
                "gender": gender,
                "emotion_code": emotion_code,
                "emotion": emotion_label,
                "split": split
            })

    df = pd.DataFrame(records)
    logger.info(f"Indexed {len(df)} speech samples across {df['actor_id'].nunique()} actors.")
    logger.info(f"Splits distribution:\n{df['split'].value_counts()}")
    return df


def build_and_cache_dataset(force: bool = False):
    """Extract and cache handcrafted feature matrices and Mel-spectrogram tensors."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    metadata_csv = CACHE_DIR / "ravdess_metadata.csv"

    if not force and metadata_csv.exists():
        logger.info("Using cached metadata...")
        df = pd.read_csv(metadata_csv)
    else:
        df = index_ravdess_dataset()
        df.to_csv(metadata_csv, index=False)

    # Label encoder: map emotion string to integer index
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(EMOTION_CLASSES)
    joblib.dump(encoder, CACHE_DIR / "label_encoder.joblib")

    # Check if features are already cached
    cache_files = [
        "X_train_handcrafted.npy", "y_train.npy",
        "X_val_handcrafted.npy", "y_val.npy",
        "X_test_handcrafted.npy", "y_test.npy",
        "X_train_mel.npy", "X_val_mel.npy", "X_test_mel.npy"
    ]
    all_cached = all((CACHE_DIR / f).exists() for f in cache_files)

    if all_cached and not force:
        logger.info("All feature matrices already cached in data/cache/. Skipping extraction.")
        return

    logger.info("Extracting features across dataset splits (this takes ~30-45 seconds)...")

    for split in ["train", "val", "test"]:
        sub_df = df[df["split"] == split].reset_index(drop=True)
        handcrafted_list = []
        mel_list = []
        labels_list = []

        for _, row in tqdm(sub_df.iterrows(), total=len(sub_df), desc=f"Processing {split}"):
            p = row["path"]
            # Handcrafted features (300 dims)
            h_feat = extract_handcrafted_features(p, subset="all")
            handcrafted_list.append(h_feat)

            # Mel-spectrogram (128, 130)
            mel = extract_mel_spectrogram(p)
            mel_list.append(mel)

            labels_list.append(row["emotion"])

        X_h = np.array(handcrafted_list, dtype=np.float32)
        X_m = np.array(mel_list, dtype=np.float32)
        y = encoder.transform(labels_list).astype(np.int64)

        np.save(CACHE_DIR / f"X_{split}_handcrafted.npy", X_h)
        np.save(CACHE_DIR / f"X_{split}_mel.npy", X_m)
        np.save(CACHE_DIR / f"y_{split}.npy", y)

        logger.info(f"Saved {split}: Handcrafted shape {X_h.shape}, Mel shape {X_m.shape}, y shape {y.shape}")

    logger.info("Feature caching complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and cache RAVDESS dataset.")
    parser.add_argument("--force", action="store_true", help="Force re-extraction of all features.")
    args = parser.parse_args()

    build_and_cache_dataset(force=args.force)
