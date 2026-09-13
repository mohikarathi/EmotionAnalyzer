"""Unit tests for audio preprocessing module."""

import numpy as np
import pytest
import soundfile as sf
import tempfile
from pathlib import Path

from src.audio.preprocessing import (
    load_audio,
    normalize_amplitude,
    pad_or_trim_audio,
    preprocess_audio,
    trim_silence,
)
from src.config import TARGET_SAMPLES, TARGET_SR


@pytest.fixture
def synthetic_wav_file():
    """Create a temporary clean 2-second 440Hz sine wave WAV file."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, signal, sr)
        tmp_path = Path(tmp.name)

    yield tmp_path

    if tmp_path.exists():
        tmp_path.unlink()


def test_normalize_amplitude():
    """Verify amplitude normalization scales peak to 1.0."""
    raw = np.array([0.0, -0.25, 0.5, -0.1], dtype=np.float32)
    norm = normalize_amplitude(raw)
    assert np.isclose(np.max(np.abs(norm)), 1.0)
    assert norm[2] == 1.0
    assert norm[1] == -0.5


def test_pad_or_trim_audio_short():
    """Verify short audio is padded symmetrically to target_samples."""
    short_signal = np.ones(1000, dtype=np.float32)
    target = 2000
    padded = pad_or_trim_audio(short_signal, target_samples=target)
    assert len(padded) == target
    assert padded[0] == 0.0
    assert padded[-1] == 0.0
    assert np.sum(padded) == 1000.0


def test_pad_or_trim_audio_long():
    """Verify long audio is center-cropped to target_samples."""
    long_signal = np.arange(3000, dtype=np.float32)
    target = 1000
    trimmed = pad_or_trim_audio(long_signal, target_samples=target)
    assert len(trimmed) == target
    assert trimmed[0] == 1000.0
    assert trimmed[-1] == 1999.0


def test_load_audio_resampling(synthetic_wav_file):
    """Verify load_audio resamples correctly and returns float32 mono."""
    target_sr = 16000
    y, sr = load_audio(synthetic_wav_file, target_sr=target_sr, mono=True)
    assert sr == target_sr
    assert y.ndim == 1
    assert y.dtype == np.float32
    assert len(y) == int(2.0 * target_sr)


def test_preprocess_audio_deterministic(synthetic_wav_file):
    """Verify end-to-end preprocessing returns exact TARGET_SAMPLES with normalized amplitude."""
    y, sr = preprocess_audio(synthetic_wav_file)
    assert sr == TARGET_SR
    assert len(y) == TARGET_SAMPLES
    assert np.max(np.abs(y)) <= 1.0001
