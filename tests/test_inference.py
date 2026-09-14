"""Unit and regression tests for model inference, probability distributions, and uncertainty."""

import numpy as np
import pytest
import soundfile as sf
import tempfile
from pathlib import Path

from src.config import CONFIDENCE_THRESHOLD, EMOTION_CLASSES
from src.models.inference import get_inference_engine


@pytest.fixture
def test_audio_file():
    """Create a temporary clean 3-second audio file."""
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.4 * np.sin(2 * np.pi * 220 * t).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, signal, sr)
        path = Path(tmp.name)

    yield path

    if path.exists():
        path.unlink()


def test_inference_engine_singleton():
    """Verify InferenceEngine enforces the singleton design pattern."""
    engine1 = get_inference_engine()
    engine2 = get_inference_engine()
    assert engine1 is engine2


def test_prediction_output_structure(test_audio_file):
    """Verify inference output format, probability sorting, and sum close to 1.0."""
    engine = get_inference_engine()
    result = engine.predict(test_audio_file, model_name="hybrid")

    assert "predicted_emotion" in result
    assert result["predicted_emotion"] in EMOTION_CLASSES
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "is_low_confidence" in result
    assert "probabilities" in result
    assert len(result["probabilities"]) == 8

    # Verify sorted descending
    scores = [item["score"] for item in result["probabilities"]]
    assert scores == sorted(scores, reverse=True)

    # Verify probability sum ≈ 1.0
    assert np.isclose(sum(scores), 1.0, atol=0.05)


def test_uncertainty_threshold_flagging(test_audio_file):
    """Verify is_low_confidence flag activates when confidence is below threshold."""
    engine = get_inference_engine()

    # Artificially high threshold should trigger low confidence flag
    res_high_thresh = engine.predict(test_audio_file, threshold=0.999)
    assert res_high_thresh["is_low_confidence"] is True

    # Artificially low threshold should NOT trigger low confidence flag
    res_low_thresh = engine.predict(test_audio_file, threshold=0.001)
    assert res_low_thresh["is_low_confidence"] is False


def test_prediction_bytesio(test_audio_file):
    """Verify inference works seamlessly with in-memory io.BytesIO audio buffers."""
    import io
    engine = get_inference_engine()

    with open(test_audio_file, "rb") as f:
        audio_bytes = f.read()

    buf = io.BytesIO(audio_bytes)
    result = engine.predict(buf, model_name="hybrid")

    assert "predicted_emotion" in result
    assert result["predicted_emotion"] in EMOTION_CLASSES
    assert 0.0 <= result["confidence"] <= 1.0

