"""Evaluation metrics and statistical reporting for Speech Emotion Recognition."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import EMOTION_CLASSES


def compute_ser_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    class_names: Optional[List[str]] = None
) -> Dict[str, Union[float, Dict, np.ndarray]]:
    """Compute comprehensive classification performance metrics.

    Calculates:
    - Overall Accuracy
    - Macro F1 (critical for SER: treats all emotion classes equally)
    - Weighted F1 (accounts for class support)
    - Macro Precision and Macro Recall
    - Per-class precision, recall, and F1
    - Confusion matrix array

    Args:
        y_true: Ground truth class indices or label strings.
        y_pred: Predicted class indices or label strings.
        class_names: Optional list of class label strings.

    Returns:
        Dictionary of computed metric values.
    """
    if class_names is None:
        class_names = EMOTION_CLASSES

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)) if isinstance(y_true[0], (int, np.integer)) else class_names)
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "confusion_matrix": cm,
        "classification_report": report
    }


def format_metrics_summary(metrics: Dict[str, Union[float, Dict]]) -> str:
    """Format evaluation metrics into a clean text summary."""
    lines = [
        f"Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)",
        f"Macro F1:        {metrics['macro_f1']:.4f}",
        f"Weighted F1:     {metrics['weighted_f1']:.4f}",
        f"Macro Precision: {metrics['macro_precision']:.4f}",
        f"Macro Recall:    {metrics['macro_recall']:.4f}",
    ]
    return "\n".join(lines)
