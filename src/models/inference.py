"""Unified model loading, inference engine, and uncertainty quantification.

Provides thread-safe, cached model management so deep learning and ML models
are loaded once into memory upon startup rather than reloaded on each HTTP request.
Maintains full backward compatibility with the original hybrid SER model.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import io

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import custom_object_scope

from src.audio.preprocessing import load_audio, normalize_amplitude, preprocess_audio
from src.config import (
    CONFIDENCE_THRESHOLD,
    EMOTION_CLASSES,
    LABEL_ENCODER_PATH,
    LEGACY_ENCODER_PATH,
    LEGACY_MODEL_PATH,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


# ---------------- Custom Keras Layers for Legacy Model ----------------
class TimeDistributedSum(Layer):
    """Custom layer computing sum across the time axis (axis=1)."""

    def call(self, inputs):
        return K.sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def get_config(self):
        return super().get_config()


class ExpandDims(Layer):
    """Custom layer expanding dimensions along a specified axis."""

    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def call(self, inputs):
        return K.expand_dims(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        return input_shape + (1,)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


# ---------------- Legacy Feature Extraction (194-d) ----------------
def extract_legacy_features(audio_path_or_data: Union[str, Path, np.ndarray], sr: Optional[int] = None) -> np.ndarray:
    """Extract the exact 194-dimensional feature vector required by the legacy model.

    Features:
    - 40 MFCC mean coefficients
    - 128 Mel-spectrogram mean coefficients
    - 7 Spectral Contrast mean coefficients
    - 12 Chroma STFT mean coefficients
    - 6 Tonnetz mean coefficients
    - 1 Harmonic component scalar (mean from HPSS)
    Total: 40 + 128 + 7 + 12 + 6 + 1 = 194 features.

    Args:
        audio_path_or_data: Filepath or numpy audio array.
        sr: Sampling rate (if numpy array passed).

    Returns:
        1D float32 array of shape (194,).
    """
    import librosa

    if isinstance(audio_path_or_data, (str, Path)) or hasattr(audio_path_or_data, "read"):
        if hasattr(audio_path_or_data, "seek"):
            audio_path_or_data.seek(0)
        y_audio, sr = librosa.load(audio_path_or_data, sr=sr)
    else:
        y_audio = audio_path_or_data
        if sr is None:
            sr = 22050

    y_audio = librosa.util.normalize(y_audio.astype(np.float32))

    mfcc = np.mean(librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=40).T, axis=0)
    mel = np.mean(librosa.power_to_db(librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128)).T, axis=0)
    spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y_audio, sr=sr).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y_audio, sr=sr).T, axis=0)
    y_harmonic, _ = librosa.effects.hpss(y_audio)
    tonnetz = np.mean(librosa.feature.tonnetz(y=y_harmonic, sr=sr).T, axis=0)
    poly = np.array([np.mean(y_harmonic)], dtype=np.float32)

    feature_vector = np.hstack([mfcc, mel, spec_contrast, chroma, tonnetz, poly]).astype(np.float32)
    return feature_vector


# ---------------- Model Registry & Singleton Cache ----------------
class InferenceEngine:
    """Singleton inference manager for loading and querying SER models."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(InferenceEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.models: Dict[str, Any] = {}
        self.label_encoder = None
        self.classes: List[str] = EMOTION_CLASSES
        self._load_default_models()
        self._initialized = True

    def _load_default_models(self):
        """Pre-load models into memory on startup."""
        # 1. Load label encoder
        if LEGACY_ENCODER_PATH.exists():
            try:
                self.label_encoder = joblib.load(LEGACY_ENCODER_PATH)
                self.classes = list(self.label_encoder.classes_)
                logger.info(f"Loaded label encoder with classes: {self.classes}")
            except Exception as e:
                logger.warning(f"Failed to load label encoder: {e}")

        # 2. Load legacy hybrid model
        if LEGACY_MODEL_PATH.exists():
            try:
                with custom_object_scope({"ExpandDims": ExpandDims, "TimeDistributedSum": TimeDistributedSum}):
                    self.models["hybrid"] = load_model(str(LEGACY_MODEL_PATH))
                logger.info("Loaded legacy Hybrid SER model successfully.")
            except Exception as e:
                logger.warning(f"Failed to load legacy hybrid model: {e}")

        # 3. Load CNN mel model if present
        cnn_path = MODELS_DIR / "cnn_mel_model.keras"
        if cnn_path.exists():
            try:
                self.models["cnn"] = load_model(str(cnn_path))
                logger.info("Loaded CNN Mel model successfully.")
            except Exception as e:
                logger.debug(f"CNN model not loaded yet: {e}")

        # 4. Load Random Forest baseline if present
        rf_path = MODELS_DIR / "classical_rf.joblib"
        if rf_path.exists():
            try:
                self.models["random_forest"] = joblib.load(str(rf_path))
                logger.info("Loaded Random Forest baseline successfully.")
            except Exception as e:
                logger.debug(f"Random Forest model not loaded yet: {e}")

        # Pre-warm JIT compilation and graph execution
        self.warmup()

    def warmup(self):
        """Pre-warm execution graphs, Numba JIT compilation, and filter caches on startup."""
        try:
            logger.info("Running inference engine warmup pass...")
            dummy_y = np.zeros(22050 // 2, dtype=np.float32)
            if "hybrid" in self.models:
                dummy_legacy = extract_legacy_features(dummy_y, sr=22050)
                self.models["hybrid"].predict(np.expand_dims(dummy_legacy, axis=(0, -1)), verbose=0)
            if "cnn" in self.models:
                from src.audio.features import extract_mel_spectrogram
                dummy_mel = extract_mel_spectrogram(dummy_y, sr=22050)
                self.models["cnn"].predict(np.expand_dims(dummy_mel, axis=(0, -1)), verbose=0)
            if "random_forest" in self.models:
                from src.audio.features import extract_handcrafted_features
                dummy_feats = extract_handcrafted_features(dummy_y, sr=22050)
                self.models["random_forest"].predict_proba(np.expand_dims(dummy_feats, axis=0))
            logger.info("Warmup complete. Latency-critical paths compiled.")
        except Exception as e:
            logger.debug(f"Inference engine warmup note: {e}")

    def predict(
        self,
        audio_input: Union[str, Path, io.BytesIO, np.ndarray],
        sr: Optional[int] = None,
        model_name: str = "hybrid",
        threshold: float = CONFIDENCE_THRESHOLD
    ) -> Dict[str, Any]:
        """Perform probability-based speech emotion prediction on an audio file or array.

        Args:
            audio_input: Audio filepath, byte stream, or preprocessed 1D numpy array.
            sr: Sampling rate (if preprocessed array passed).
            model_name: Name of model to query ('hybrid', 'cnn', 'random_forest').
            threshold: Confidence threshold for uncertainty quantification.

        Returns:
            Dictionary containing prediction results, probabilities, and model metadata.
        """
        # Determine available model fallback
        if model_name not in self.models:
            if "hybrid" in self.models:
                model_name = "hybrid"
            elif self.models:
                model_name = next(iter(self.models.keys()))
            else:
                raise RuntimeError("No trained model is currently available in the inference engine.")

        model = self.models[model_name]

        # Extract appropriate representation
        if model_name in ("hybrid", "legacy"):
            features = extract_legacy_features(audio_input, sr=sr)
            tensor_input = np.expand_dims(features, axis=(0, -1))  # (1, 194, 1)
            raw_preds = model.predict(tensor_input, verbose=0)[0]
        elif model_name == "cnn":
            from src.audio.features import extract_mel_spectrogram
            mel_spec = extract_mel_spectrogram(audio_input, sr=sr or 22050)
            tensor_input = np.expand_dims(mel_spec, axis=(0, -1))  # (1, 128, T, 1)
            raw_preds = model.predict(tensor_input, verbose=0)[0]
        elif model_name in ("random_forest", "classical"):
            from src.audio.features import extract_handcrafted_features
            features = extract_handcrafted_features(audio_input, sr=sr or 22050)
            raw_preds = model.predict_proba(np.expand_dims(features, axis=0))[0]
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        # Convert to standard Python floats and build probability distribution
        raw_preds = np.asarray(raw_preds, dtype=np.float64)
        # Numerical normalization in case model outputs didn't sum to exactly 1.0
        sum_preds = np.sum(raw_preds)
        if sum_preds > 0:
            probs = raw_preds / sum_preds
        else:
            probs = np.ones_like(raw_preds) / len(raw_preds)

        # Map to class labels
        class_names = self.classes
        if len(probs) != len(class_names):
            class_names = EMOTION_CLASSES[:len(probs)]

        prob_list = [
            {
                "emotion": class_name,
                "score": float(np.round(prob, 4)),
                "percentage": float(np.round(prob * 100, 1))
            }
            for class_name, prob in zip(class_names, probs)
        ]

        # Sort descending by probability score
        prob_list.sort(key=lambda item: item["score"], reverse=True)

        top_pred = prob_list[0]
        confidence = top_pred["score"]
        is_low_conf = bool(confidence < threshold)

        return {
            "predicted_emotion": top_pred["emotion"],
            "confidence": confidence,
            "is_low_confidence": is_low_conf,
            "confidence_threshold": threshold,
            "probabilities": prob_list,
            "model_used": model_name
        }


# Global helper instance
_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    """Obtain or initialize the global singleton inference engine."""
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine


def predict_emotion(audio_path: Union[str, Path]) -> str:
    """Backward-compatible wrapper returning predicted emotion string."""
    engine = get_inference_engine()
    result = engine.predict(audio_path)
    return result["predicted_emotion"]
