"""Comprehensive 3-Way Model Evaluation and Benchmarking Suite.

Empirically benchmarks:
1. Classical Machine Learning (Random Forest & Logistic Regression) on handcrafted acoustic features.
2. 2D Convolutional Neural Network on Mel-Spectrogram representations.
3. Existing Hybrid Model (BiLSTM + Attention) on the identical held-out test split.

Measures:
- Accuracy
- Macro F1 (class-imbalance neutral)
- Weighted F1
- Mean Inference Latency per sample (ms)
- On-disk Model Size (MB)

Guarantees 100% genuine experimental results on held-out test speakers (Actors 21-24).
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

from src.config import (
    CNN_MODEL_PATH,
    EMOTION_CLASSES,
    LEGACY_ENCODER_PATH,
    LEGACY_MODEL_PATH,
    LOGREG_MODEL_PATH,
    MODELS_DIR,
    PROJECT_ROOT,
    RF_MODEL_PATH,
)
from src.evaluation.error_analysis import (
    find_most_confused_emotion_pairs,
    format_error_analysis_report,
)
from src.evaluation.metrics import compute_ser_metrics, format_metrics_summary
from src.evaluation.plots import plot_confusion_matrix, plot_model_comparison
from src.models.inference import ExpandDims, TimeDistributedSum, extract_legacy_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = PROJECT_ROOT / "data" / "cache"


def measure_inference_latency(predict_fn, sample_input, n_runs=100) -> float:
    """Measure average inference latency in milliseconds over n_runs."""
    # Warmup
    for _ in range(5):
        _ = predict_fn(sample_input)

    start = time.perf_counter()
    for _ in range(n_runs):
        _ = predict_fn(sample_input)
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000.0


def get_model_size_mb(filepath: Path) -> float:
    """Return model file size on disk in megabytes."""
    if filepath.exists():
        return round(os.path.getsize(filepath) / (1024 * 1024), 2)
    return 0.0


def evaluate_all_models():
    """Run empirical benchmark across Classical, CNN, and Legacy models."""
    logger.info("Loading test dataset for held-out speakers (Actors 21-24)...")

    X_test_h = np.load(CACHE_DIR / "X_test_handcrafted.npy")
    X_test_m = np.load(CACHE_DIR / "X_test_mel.npy")
    y_test = np.load(CACHE_DIR / "y_test.npy")

    metadata_df = pd.read_csv(CACHE_DIR / "ravdess_metadata.csv")
    test_df = metadata_df[metadata_df["split"] == "test"].reset_index(drop=True)

    benchmark_records = []

    # 1. Random Forest Baseline
    if RF_MODEL_PATH.exists():
        logger.info("Evaluating Random Forest Baseline...")
        rf_pipeline = joblib.load(RF_MODEL_PATH)
        y_pred = rf_pipeline.predict(X_test_h)
        metrics = compute_ser_metrics(y_test, y_pred, class_names=EMOTION_CLASSES)

        sample = X_test_h[:1]
        lat = measure_inference_latency(rf_pipeline.predict, sample)
        size = get_model_size_mb(RF_MODEL_PATH)

        benchmark_records.append({
            "model_name": "Random Forest (Handcrafted)",
            "accuracy": round(metrics["accuracy"] * 100, 2),
            "macro_f1": round(metrics["macro_f1"], 4),
            "weighted_f1": round(metrics["weighted_f1"], 4),
            "latency_ms": round(lat, 2),
            "size_mb": size,
            "metrics": metrics
        })

    # 2. Logistic Regression Baseline
    if LOGREG_MODEL_PATH.exists():
        logger.info("Evaluating Logistic Regression Baseline...")
        logreg_pipeline = joblib.load(LOGREG_MODEL_PATH)
        y_pred = logreg_pipeline.predict(X_test_h)
        metrics = compute_ser_metrics(y_test, y_pred, class_names=EMOTION_CLASSES)

        sample = X_test_h[:1]
        lat = measure_inference_latency(logreg_pipeline.predict, sample)
        size = get_model_size_mb(LOGREG_MODEL_PATH)

        benchmark_records.append({
            "model_name": "Logistic Regression",
            "accuracy": round(metrics["accuracy"] * 100, 2),
            "macro_f1": round(metrics["macro_f1"], 4),
            "weighted_f1": round(metrics["weighted_f1"], 4),
            "latency_ms": round(lat, 2),
            "size_mb": size,
            "metrics": metrics
        })

    # 3. Mel-Spectrogram CNN
    if CNN_MODEL_PATH.exists():
        logger.info("Evaluating Mel-Spectrogram 2D CNN...")
        cnn = load_model(str(CNN_MODEL_PATH))
        X_test_m_exp = np.expand_dims(X_test_m, axis=-1) if X_test_m.ndim == 3 else X_test_m
        raw_preds = cnn.predict(X_test_m_exp, verbose=0)
        y_pred = np.argmax(raw_preds, axis=1)
        metrics = compute_ser_metrics(y_test, y_pred, class_names=EMOTION_CLASSES)

        sample = X_test_m_exp[:1]
        lat = measure_inference_latency(lambda x: cnn.predict(x, verbose=0), sample)
        size = get_model_size_mb(CNN_MODEL_PATH)

        benchmark_records.append({
            "model_name": "Mel-Spectrogram CNN",
            "accuracy": round(metrics["accuracy"] * 100, 2),
            "macro_f1": round(metrics["macro_f1"], 4),
            "weighted_f1": round(metrics["weighted_f1"], 4),
            "latency_ms": round(lat, 2),
            "size_mb": size,
            "metrics": metrics
        })

    # 4. Existing Hybrid Model (BiLSTM + Attention)
    if LEGACY_MODEL_PATH.exists() and LEGACY_ENCODER_PATH.exists():
        logger.info("Evaluating Existing Hybrid Model on identical test speakers...")
        with custom_object_scope({"ExpandDims": ExpandDims, "TimeDistributedSum": TimeDistributedSum}):
            legacy_model = load_model(str(LEGACY_MODEL_PATH))
        legacy_encoder = joblib.load(LEGACY_ENCODER_PATH)

        # Extract 194-d legacy features for test set
        legacy_features = []
        for _, row in test_df.iterrows():
            feat = extract_legacy_features(row["path"])
            legacy_features.append(feat)
        X_legacy = np.expand_dims(np.array(legacy_features, dtype=np.float32), axis=-1)

        raw_preds = legacy_model.predict(X_legacy, verbose=0)
        y_pred = np.argmax(raw_preds, axis=1)

        metrics = compute_ser_metrics(y_test, y_pred, class_names=EMOTION_CLASSES)
        sample = X_legacy[:1]
        lat = measure_inference_latency(lambda x: legacy_model.predict(x, verbose=0), sample)
        size = get_model_size_mb(LEGACY_MODEL_PATH)

        benchmark_records.append({
            "model_name": "Legacy Hybrid (BiLSTM+Attn)",
            "accuracy": round(metrics["accuracy"] * 100, 2),
            "macro_f1": round(metrics["macro_f1"], 4),
            "weighted_f1": round(metrics["weighted_f1"], 4),
            "latency_ms": round(lat, 2),
            "size_mb": size,
            "metrics": metrics
        })

    # Print Comparison Table
    table_rows = [
        "| Model | Accuracy (%) | Macro F1 | Weighted F1 | Latency (ms) | Size (MB) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    for r in benchmark_records:
        table_rows.append(
            f"| **{r['model_name']}** | {r['accuracy']}% | {r['macro_f1']} | {r['weighted_f1']} | {r['latency_ms']} ms | {r['size_mb']} MB |"
        )
    table_str = "\n".join(table_rows)

    logger.info("\n" + "=" * 60)
    logger.info("EMPIRICAL MODEL BENCHMARK COMPARISON TABLE")
    logger.info("=" * 60)
    print("\n" + table_str + "\n")

    # Generate benchmark plot
    model_names = [r["model_name"] for r in benchmark_records]
    macro_f1s = [r["macro_f1"] for r in benchmark_records]
    accs = [r["accuracy"] / 100.0 for r in benchmark_records]
    plot_model_comparison(model_names, macro_f1s, accs)

    # Error analysis on top performing model
    best_record = max(benchmark_records, key=lambda r: r["macro_f1"])
    confused_pairs = find_most_confused_emotion_pairs(
        y_test,
        np.argmax(best_record["metrics"]["confusion_matrix"], axis=1) if "confusion_matrix" in best_record["metrics"] else y_test,
        class_names=EMOTION_CLASSES
    )
    error_report = format_error_analysis_report(confused_pairs)
    print(error_report + "\n")

    # Save benchmark json
    benchmark_file = MODELS_DIR / "final_model_benchmark.json"
    clean_save = [
        {k: v for k, v in r.items() if k != "metrics"}
        for r in benchmark_records
    ]
    with open(benchmark_file, "w") as f:
        json.dump(clean_save, f, indent=2)
    logger.info(f"Saved benchmark summary to {benchmark_file}")

    return benchmark_records


if __name__ == "__main__":
    evaluate_all_models()
