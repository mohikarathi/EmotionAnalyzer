"""Unified CLI training script for Speech Emotion Recognition models.

Supports training:
- Classical ML baselines: Random Forest, Logistic Regression, SVC
- 2D Mel-Spectrogram Convolutional Neural Network
- Complete suite comparison on strict speaker-independent test splits

Usage:
  python scripts/train.py --model classical
  python scripts/train.py --model cnn
  python scripts/train.py --model all
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np

from src.config import (
    CNN_MODEL_PATH,
    EMOTION_CLASSES,
    LOGREG_MODEL_PATH,
    MODELS_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    RF_MODEL_PATH,
)
from src.evaluation.metrics import compute_ser_metrics, format_metrics_summary
from src.evaluation.plots import plot_confusion_matrix
from src.models.classical import (
    load_classical_model,
    save_classical_model,
    train_classical_model,
)
from src.models.cnn import train_mel_cnn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = PROJECT_ROOT / "data" / "cache"


def load_cached_data():
    """Load pre-extracted train, validation, and test arrays from data/cache/."""
    if not (CACHE_DIR / "X_train_handcrafted.npy").exists():
        from scripts.prepare_data import build_and_cache_dataset
        logger.info("Cached features not found. Building dataset cache first...")
        build_and_cache_dataset()

    X_train_h = np.load(CACHE_DIR / "X_train_handcrafted.npy")
    y_train = np.load(CACHE_DIR / "y_train.npy")

    X_val_h = np.load(CACHE_DIR / "X_val_handcrafted.npy")
    y_val = np.load(CACHE_DIR / "y_val.npy")

    X_test_h = np.load(CACHE_DIR / "X_test_handcrafted.npy")
    y_test = np.load(CACHE_DIR / "y_test.npy")

    X_train_m = np.load(CACHE_DIR / "X_train_mel.npy")
    X_val_m = np.load(CACHE_DIR / "X_val_mel.npy")
    X_test_m = np.load(CACHE_DIR / "X_test_mel.npy")

    return {
        "X_train_h": X_train_h,
        "y_train": y_train,
        "X_val_h": X_val_h,
        "y_val": y_val,
        "X_test_h": X_test_h,
        "y_test": y_test,
        "X_train_m": X_train_m,
        "X_val_m": X_val_m,
        "X_test_m": X_test_m
    }


def train_and_eval_classical(data, model_type="random_forest", save_path=RF_MODEL_PATH):
    """Train and evaluate a classical ML pipeline."""
    logger.info(f"\n==================== Training {model_type.upper()} ====================")
    # Combine train + val for classical fitting (standard practice for tabular pipelines)
    X_tr = np.vstack([data["X_train_h"], data["X_val_h"]])
    y_tr = np.concatenate([data["y_train"], data["y_val"]])
    X_te = data["X_test_h"]
    y_te = data["y_test"]

    pipeline = train_classical_model(X_tr, y_tr, model_type=model_type)
    save_classical_model(pipeline, save_path)

    # Evaluate on held-out test speakers (Actors 21-24)
    y_pred = pipeline.predict(X_te)
    metrics = compute_ser_metrics(y_te, y_pred, class_names=EMOTION_CLASSES)

    logger.info(f"\n--- {model_type.upper()} Test Set Evaluation Results (Speakers 21-24) ---")
    logger.info("\n" + format_metrics_summary(metrics))

    # Plot confusion matrix
    cm_path = PROJECT_ROOT / "static" / "img" / f"cm_{model_type}.png"
    plot_confusion_matrix(metrics["confusion_matrix"], save_path=cm_path, title=f"{model_type.replace('_', ' ').title()} Confusion Matrix")

    return metrics


def train_and_eval_cnn(data, epochs=40, batch_size=32):
    """Train and evaluate the Mel-Spectrogram CNN."""
    logger.info("\n==================== Training Mel-Spectrogram CNN ====================")
    X_tr = data["X_train_m"]
    y_tr = data["y_train"]
    X_va = data["X_val_m"]
    y_va = data["y_val"]
    X_te = data["X_test_m"]
    y_te = data["y_test"]

    model, _ = train_mel_cnn(
        X_tr, y_tr, X_va, y_va,
        save_path=CNN_MODEL_PATH,
        epochs=epochs,
        batch_size=batch_size
    )

    # Predict on test set
    if X_te.ndim == 3:
        X_te_expanded = np.expand_dims(X_te, axis=-1)
    else:
        X_te_expanded = X_te

    raw_preds = model.predict(X_te_expanded)
    y_pred = np.argmax(raw_preds, axis=1)

    metrics = compute_ser_metrics(y_te, y_pred, class_names=EMOTION_CLASSES)

    logger.info("\n--- Mel-Spectrogram CNN Test Set Evaluation Results (Speakers 21-24) ---")
    logger.info("\n" + format_metrics_summary(metrics))

    cm_path = PROJECT_ROOT / "static" / "img" / "cm_cnn.png"
    plot_confusion_matrix(metrics["confusion_matrix"], save_path=cm_path, title="Mel-Spectrogram CNN Confusion Matrix")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train EmotionAnalyzer Speech Emotion Models")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "classical", "random_forest", "logistic_regression", "svc", "cnn"],
        help="Which model to train."
    )
    parser.add_argument("--epochs", type=int, default=35, help="CNN training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="CNN batch size.")
    args = parser.parse_args()

    data = load_cached_data()
    results = {}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.model in ("all", "classical", "random_forest"):
        results["random_forest"] = train_and_eval_classical(data, "random_forest", RF_MODEL_PATH)

    if args.model in ("all", "classical", "logistic_regression"):
        results["logistic_regression"] = train_and_eval_classical(data, "logistic_regression", LOGREG_MODEL_PATH)

    if args.model in ("all", "classical", "svc"):
        svc_path = MODELS_DIR / "classical_svc.joblib"
        results["svc"] = train_and_eval_classical(data, "svc", svc_path)

    if args.model in ("all", "cnn"):
        results["cnn"] = train_and_eval_cnn(data, epochs=args.epochs, batch_size=args.batch_size)

    # Save metrics summary to JSON
    summary_path = MODELS_DIR / "training_benchmark_results.json"
    serializable_results = {
        name: {
            k: v for k, v in m.items() if k != "confusion_matrix"
        }
        for name, m in results.items()
    }
    with open(summary_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"\nAll training tasks complete! Benchmark results written to {summary_path}")


if __name__ == "__main__":
    main()
