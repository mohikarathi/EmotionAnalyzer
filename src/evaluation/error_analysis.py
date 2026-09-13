"""Acoustic Error Analysis and Confusion Diagnostics for Speech Emotion Recognition."""

from typing import Dict, List, Tuple

import numpy as np

from src.config import EMOTION_CLASSES


# Domain knowledge dictionary explaining acoustic overlaps between emotion pairs
ACOUSTIC_CONFUSION_RATIONALE = {
    ("sad", "neutral"): (
        "Both Sadness and Neutral speech exhibit low fundamental frequency (F0) variance, "
        "subdued acoustic energy, and flat intonation contours, making them close in spectral centroid and RMS space."
    ),
    ("neutral", "sad"): (
        "Neutral speech is frequently confused with Sadness when actors speak slowly or with low vocal effort, "
        "resulting in similar harmonic-to-noise ratios and low spectral rolloff."
    ),
    ("angry", "happy"): (
        "Anger and Happiness are both high-arousal emotions characterized by elevated mean F0, "
        "expanded dynamic pitch range, high vocal intensity, and rapid speech cadence. "
        "Their spectral envelopes overlap significantly despite divergent emotional valence."
    ),
    ("happy", "angry"): (
        "High vocal excitement in Happiness produces sharp formant onsets and high energy across upper harmonics, "
        "frequently triggering high-arousal acoustic features shared with Anger."
    ),
    ("fearful", "surprised"): (
        "Fear and Surprise both feature abrupt pitch jumps, increased jitter, and high spectral tilt. "
        "Without linguistic context, brief acoustic bursts of surprise closely mirror acoustic startle/fear reflexes."
    ),
    ("surprised", "fearful"): (
        "Surprise is frequently classified as Fear due to rapid fundamental frequency inflection "
        "and elevated vocal tract brightness (high spectral centroid)."
    ),
    ("calm", "neutral"): (
        "Calmness and Neutrality occupy adjacent coordinates in the circumplex model of affect "
        "(low arousal, moderate valence). Both exhibit smooth energy decays and standard formant spacing."
    ),
    ("disgust", "sad"): (
        "Disgust often manifests with creaky phonation, lower pitch, and extended vowel durations, "
        "which acoustic classifiers can conflate with the sluggish temporal dynamics of Sadness."
    )
}


def find_most_confused_emotion_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = EMOTION_CLASSES,
    top_k: int = 5
) -> List[Dict[str, any]]:
    """Identify the top-K most frequently confused pairs of emotions where y_true != y_pred.

    Args:
        y_true: Array of true class indices.
        y_pred: Array of predicted class indices.
        class_names: List of class emotion names.
        top_k: Number of confused pairs to return.

    Returns:
        List of dictionaries with true_emotion, predicted_emotion, count, rate, and acoustic explanation.
    """
    confusion_counts = {}

    for t, p in zip(y_true, y_pred):
        if t != p:
            pair = (class_names[t], class_names[p])
            confusion_counts[pair] = confusion_counts.get(pair, 0) + 1

    sorted_pairs = sorted(confusion_counts.items(), key=lambda item: item[1], reverse=True)

    results = []
    total_errors = sum(confusion_counts.values())

    for (true_emo, pred_emo), count in sorted_pairs[:top_k]:
        error_pct = (count / total_errors) * 100 if total_errors > 0 else 0
        rationale = ACOUSTIC_CONFUSION_RATIONALE.get(
            (true_emo, pred_emo),
            f"Acoustic feature overlap in vocal tract resonance, pitch modulation, and spectral centroid between {true_emo} and {pred_emo}."
        )

        results.append({
            "true_emotion": true_emo,
            "predicted_emotion": pred_emo,
            "error_count": int(count),
            "error_percentage": round(error_pct, 1),
            "acoustic_explanation": rationale
        })

    return results


def format_error_analysis_report(confused_pairs: List[Dict[str, any]]) -> str:
    """Format identified confusion pairs into a readable markdown report."""
    lines = [
        "### Top Confused Emotion Pairs (Acoustic Diagnostic Analysis)\n",
        "| True Emotion | Mispredicted As | Error Count | % of All Errors | Acoustic Diagnostic Explanation |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]
    for p in confused_pairs:
        lines.append(
            f"| **{p['true_emotion'].capitalize()}** | **{p['predicted_emotion'].capitalize()}** | "
            f"{p['error_count']} | {p['error_percentage']}% | {p['acoustic_explanation']} |"
        )
    return "\n".join(lines)
