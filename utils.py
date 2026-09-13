"""Backward-compatibility wrapper for EmotionAnalyzer.

Re-exports core inference functions and custom layers so that existing
scripts, notebooks, and benchmarks continue to function without modification.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

from src.models.inference import (
    ExpandDims,
    TimeDistributedSum,
    extract_legacy_features as extract_audio_features,
    get_inference_engine,
    predict_emotion as _engine_predict_emotion,
)
from src.config import LEGACY_ENCODER_PATH, LEGACY_MODEL_PATH

# Expose model paths and loaded references for legacy consumers
MODEL_PATH = str(LEGACY_MODEL_PATH)
ENCODER_PATH = str(LEGACY_ENCODER_PATH)

_engine = get_inference_engine()
model = _engine.models.get("hybrid", None)
encoder = _engine.label_encoder


def predict_emotion(audio_path):
    """Legacy interface: returns predicted emotion label as a string."""
    try:
        return _engine_predict_emotion(audio_path)
    except Exception as e:
        return f"Error: {str(e)}"
