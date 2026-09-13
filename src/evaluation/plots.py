"""Plotting and visualization utilities for evaluation and error analysis."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import EMOTION_CLASSES, STATIC_DIR


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Acoustic Emotion Confusion Matrix",
    normalize: bool = True
) -> Path:
    """Plot and save a high-contrast, publication-grade confusion matrix heatmap.

    Args:
        cm: 2D confusion matrix array.
        class_names: List of class emotion names.
        save_path: Destination path for saved image.
        title: Title of plot.
        normalize: If True, normalize cell values by true class support (percentages).

    Returns:
        Path to saved PNG plot.
    """
    if class_names is None:
        class_names = [e.capitalize() for e in EMOTION_CLASSES]
    else:
        class_names = [e.capitalize() for e in class_names]

    if save_path is None:
        save_path = STATIC_DIR / "img" / "confusion_matrix.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        fmt = ".2f"
        data_to_plot = cm_norm
    else:
        fmt = "d"
        data_to_plot = cm

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    sns.heatmap(
        data_to_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
        ax=ax,
        linewidths=0.5,
        linecolor="#1E293B"
    )

    ax.set_title(title, color="#F8FAFC", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Acoustic Emotion", color="#94A3B8", fontsize=10, labelpad=8)
    ax.set_ylabel("True Ground Truth Emotion", color="#94A3B8", fontsize=10, labelpad=8)

    ax.tick_params(colors="#94A3B8", labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors="#94A3B8", labelsize=8)
    cbar.ax.yaxis.label.set_color("#94A3B8")

    plt.tight_layout()
    fig.savefig(str(save_path), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_model_comparison(
    models: List[str],
    macro_f1s: List[float],
    accuracies: List[float],
    save_path: Optional[Path] = None
) -> Path:
    """Generate a clean grouped bar chart comparing multiple models."""
    if save_path is None:
        save_path = STATIC_DIR / "img" / "model_comparison.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    rects1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="#3B82F6")
    rects2 = ax.bar(x + width / 2, macro_f1s, width, label="Macro F1", color="#6366F1")

    ax.set_ylabel("Score", color="#94A3B8", fontsize=10)
    ax.set_title("Speech Emotion Recognition Model Benchmark", color="#F8FAFC", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, color="#94A3B8", fontsize=9)
    ax.tick_params(colors="#94A3B8")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.15, color="#FFFFFF")

    legend = ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

    # Add values on top of bars
    for rect in list(rects1) + list(rects2):
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#E2E8F0",
            fontsize=8
        )

    plt.tight_layout()
    fig.savefig(str(save_path), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return save_path
