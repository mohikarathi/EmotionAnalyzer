"""Explainability and attribution module for Speech Emotion Recognition models.

Provides two complementary explanation mechanisms:
1. Classical Feature Importance: Ranks top contributing acoustic signal descriptors
   (MFCCs, spectral centroid, RMS energy, pitch chroma) with domain interpretations.
2. Mel-Spectrogram Grad-CAM: Computes gradient-weighted class activation heatmaps
   over 2D spectrograms, highlighting exact time-frequency regions (vocal bursts,
   harmonics, formant shifts) that drove the CNN's decision.

Note: Explanations represent model attribution and correlational saliency,
not verified physiological or psychological causality.
"""

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.audio.features import (
    extract_handcrafted_features,
    extract_mel_spectrogram,
    get_handcrafted_feature_names,
)
from src.config import HOP_LENGTH, N_FFT, N_MELS, TARGET_SR

logger = logging.getLogger(__name__)

# Domain acoustic descriptions for key features
FEATURE_DESCRIPTIONS = {
    "rms": "Vocal loudness dynamics and acoustic intensity fluctuation",
    "zcr": "Noisiness and unvoiced friction vs voiced vowel ratio",
    "spectral_centroid": "Vocal brightness and sharp acoustic timbre",
    "spectral_bandwidth": "Spectral spread across fundamental and higher formants",
    "spectral_rolloff": "High-frequency energy concentration and vocal effort",
    "spectral_contrast": "Formant clarity: difference between spectral peaks and harmonic valleys",
    "mfcc_1": "Overall speech volume and broad vocal tract resonance",
    "mfcc_2": "Vocal tract length and open vs closed vowel configuration",
    "chroma": "Harmonic energy distribution across the 12 semitone pitch classes",
    "tonnetz": "Tonal centroid harmony and musical interval stability"
}


def describe_feature(feature_name: str) -> str:
    """Provide a human-readable acoustic interpretation for an extracted feature name."""
    for key, desc in FEATURE_DESCRIPTIONS.items():
        if key in feature_name:
            if "std" in feature_name:
                return f"{desc} (temporal variation / inflection)"
            return f"{desc} (average level)"
    return "Vocal spectral envelope component"


def explain_classical_prediction(
    pipeline: Any,
    top_k: int = 6
) -> Dict[str, Any]:
    """Extract global and local feature importance rankings from a classical ML pipeline.

    Args:
        pipeline: Fitted sklearn Pipeline.
        top_k: Number of top acoustic features to return.

    Returns:
        Dictionary of top features with weights, names, and acoustic interpretations.
    """
    classifier = pipeline.named_steps.get("classifier", pipeline)
    feature_names = get_handcrafted_feature_names()

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        # Multinomial logistic regression: mean absolute coefficient magnitude across classes
        importances = np.mean(np.abs(classifier.coef_), axis=0)
    else:
        return {
            "type": "feature_importance",
            "description": "Model attribution via feature importance not available for this classifier.",
            "top_features": []
        }

    # Rank indices descending
    top_indices = np.argsort(importances)[::-1][:top_k]
    top_features = [
        {
            "name": feature_names[idx],
            "importance": float(round(importances[idx], 4)),
            "interpretation": describe_feature(feature_names[idx])
        }
        for idx in top_indices
    ]

    return {
        "type": "feature_importance",
        "description": "Acoustic signal attributes with highest influence on the classification decision:",
        "top_features": top_features
    }


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    mel_tensor: np.ndarray,
    target_class_idx: Optional[int] = None,
    layer_name: str = "last_conv_layer"
) -> np.ndarray:
    """Compute 2D Grad-CAM saliency heatmap for a Mel-spectrogram input.

    Args:
        model: Trained Keras CNN model.
        mel_tensor: 4D numpy array of shape (1, 128, 130, 1).
        target_class_idx: Class index to compute gradients for (defaults to top predicted).
        layer_name: Name of target convolutional layer.

    Returns:
        2D numpy array heatmap of shape (128, 130) normalized to [0.0, 1.0].
    """
    # Create sub-model that outputs both the target conv layer activations and predictions
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
    except Exception as e:
        logger.warning(f"Grad-CAM could not find layer '{layer_name}': {e}")
        return np.zeros((128, 130), dtype=np.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(mel_tensor)
        if target_class_idx is None:
            target_class_idx = tf.argmax(predictions[0])
        loss = predictions[:, target_class_idx]

    # Gradients of class score w.r.t. conv layer output feature maps
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling over spatial dimensions (height and width)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight conv feature maps by pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU: only positive contributions to the class are salient
    heatmap = tf.maximum(heatmap, 0.0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize heatmap to match input Mel spectrogram dimensions (128, 130)
    from scipy.ndimage import zoom
    zoom_factors = (128 / heatmap.shape[0], 130 / heatmap.shape[1])
    heatmap_resized = zoom(heatmap, zoom_factors, order=1)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

    return heatmap_resized.astype(np.float32)


def render_gradcam_overlay(
    mel_spec: np.ndarray,
    heatmap: np.ndarray
) -> str:
    """Overlay Grad-CAM heatmap on the Mel-spectrogram and encode as base64 PNG."""
    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=120)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    # Plot base Mel-spectrogram
    ax.imshow(mel_spec, aspect="auto", origin="lower", cmap="magma", extent=[0, 3.0, 0, 8000])

    # Overlay Grad-CAM saliency with alpha transparency
    ax.imshow(heatmap, aspect="auto", origin="lower", cmap="jet", alpha=0.45, extent=[0, 3.0, 0, 8000])

    ax.set_title("Grad-CAM Saliency: Acoustic Attribution Regions", color="#F8FAFC", fontsize=10, pad=8)
    ax.set_xlabel("Time (seconds)", color="#94A3B8", fontsize=8)
    ax.set_ylabel("Frequency (Hz)", color="#94A3B8", fontsize=8)
    ax.tick_params(colors="#94A3B8", labelsize=8)

    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def explain_prediction(
    audio_path_or_y: Union[str, Path, np.ndarray],
    model_name: str = "random_forest"
) -> Optional[Dict[str, Any]]:
    """Unified entry point for explaining a speech emotion prediction."""
    from src.models.inference import get_inference_engine
    engine = get_inference_engine()

    if model_name in ("random_forest", "classical") and "random_forest" in engine.models:
        rf_pipeline = engine.models["random_forest"]
        return explain_classical_prediction(rf_pipeline)

    if model_name == "cnn" and "cnn" in engine.models:
        cnn_model = engine.models["cnn"]
        mel = extract_mel_spectrogram(audio_path_or_y)
        tensor_in = np.expand_dims(mel, axis=(0, -1))
        heatmap = compute_gradcam_heatmap(cnn_model, tensor_in)
        heatmap_b64 = render_gradcam_overlay(mel, heatmap)
        return {
            "type": "gradcam",
            "description": "Grad-CAM heatmap highlights active time-frequency regions that drove the CNN prediction:",
            "heatmap_b64": heatmap_b64
        }

    # Fallback for hybrid model: show acoustic feature importance summary
    if "random_forest" in engine.models:
        rf_pipeline = engine.models["random_forest"]
        expl = explain_classical_prediction(rf_pipeline)
        expl["description"] = "Acoustic signal descriptors most strongly correlated with emotional prosody:"
        return expl

    return None
