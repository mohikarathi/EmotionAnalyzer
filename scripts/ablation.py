"""Acoustic Feature Ablation Study for Speech Emotion Recognition.

Empirically evaluates the incremental value of distinct acoustic representations:
Experiment 1: MFCCs Only (40 MFCCs + First Derivatives + Second Derivatives = 240 dims)
Experiment 2: MFCCs + Spectral Features (+ ZCR, RMS, Centroid, Bandwidth, Rolloff, Contrast = 264 dims)
Experiment 3: Full Feature Set (+ Chroma Pitch Classes + Tonnetz Harmonics = 300 dims)

Answers the scientific research question:
'Which feature groups actually contribute to emotion classification generalization?'
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import EMOTION_CLASSES, MODELS_DIR, PROJECT_ROOT, RANDOM_SEED
from src.evaluation.metrics import compute_ser_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = PROJECT_ROOT / "data" / "cache"


def run_ablation_experiments():
    logger.info("Loading cached feature datasets...")
    X_train_full = np.load(CACHE_DIR / "X_train_handcrafted.npy")
    y_train = np.load(CACHE_DIR / "y_train.npy")

    X_val_full = np.load(CACHE_DIR / "X_val_handcrafted.npy")
    y_val = np.load(CACHE_DIR / "y_val.npy")

    X_test_full = np.load(CACHE_DIR / "X_test_handcrafted.npy")
    y_test = np.load(CACHE_DIR / "y_test.npy")

    # Combine train + val for classical fitting
    X_train_combined = np.vstack([X_train_full, X_val_full])
    y_train_combined = np.concatenate([y_train, y_val])

    # Feature index slices based on extract_handcrafted_features:
    # 0 to 240: MFCCs + deltas + delta2
    # 240 to 264: ZCR(2) + RMS(2) + Centroid(2) + Bandwidth(2) + Rolloff(2) + Contrast(14) = 24 dims
    # 264 to 300: Chroma(24) + Tonnetz(12) = 36 dims
    experiments = [
        {
            "name": "Experiment 1: MFCC Only",
            "features_used": "40 MFCCs + Deltas + Delta-Deltas (mean & std)",
            "dimension": 240,
            "train_slice": X_train_combined[:, :240],
            "test_slice": X_test_full[:, :240]
        },
        {
            "name": "Experiment 2: MFCC + Spectral",
            "features_used": "MFCCs + ZCR + RMS + Centroid + Bandwidth + Rolloff + Contrast",
            "dimension": 264,
            "train_slice": X_train_combined[:, :264],
            "test_slice": X_test_full[:, :264]
        },
        {
            "name": "Experiment 3: Full Feature Set",
            "features_used": "Full Acoustic Set (+ Chroma Pitch Classes + Tonnetz Tonal Centroids)",
            "dimension": 300,
            "train_slice": X_train_combined,
            "test_slice": X_test_full
        }
    ]

    results = []

    for exp in experiments:
        logger.info(f"Running {exp['name']} ({exp['dimension']} features)...")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=16,
                min_samples_split=4,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                class_weight="balanced"
            ))
        ])

        pipeline.fit(exp["train_slice"], y_train_combined)
        y_pred = pipeline.predict(exp["test_slice"])
        metrics = compute_ser_metrics(y_test, y_pred, class_names=EMOTION_CLASSES)

        record = {
            "experiment": exp["name"],
            "features": exp["features_used"],
            "dimension": exp["dimension"],
            "accuracy": round(metrics["accuracy"] * 100, 2),
            "macro_f1": round(metrics["macro_f1"], 4),
            "weighted_f1": round(metrics["weighted_f1"], 4)
        }
        results.append(record)
        logger.info(f"{exp['name']} -> Accuracy: {record['accuracy']}%, Macro F1: {record['macro_f1']}")

    # Print Table
    table_lines = [
        "| Experiment | Feature Configuration | Dimension | Accuracy (%) | Macro F1 | Weighted F1 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    for r in results:
        table_lines.append(
            f"| **{r['experiment']}** | {r['features']} | {r['dimension']} | {r['accuracy']}% | **{r['macro_f1']}** | {r['weighted_f1']} |"
        )
    table_str = "\n".join(table_lines)

    logger.info("\n" + "=" * 60)
    logger.info("FEATURE ABLATION STUDY RESULTS")
    logger.info("=" * 60)
    print("\n" + table_str + "\n")

    # Save to JSON
    out_file = MODELS_DIR / "ablation_study.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved ablation study to {out_file}")

    return results


if __name__ == "__main__":
    run_ablation_experiments()
