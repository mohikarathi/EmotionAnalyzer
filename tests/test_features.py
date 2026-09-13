"""Unit tests for feature engineering and extraction modules."""

import numpy as np
import pytest

from src.audio.features import (
    extract_handcrafted_features,
    extract_mel_spectrogram,
    get_handcrafted_feature_names,
)
from src.audio.preprocessing import preprocess_audio
from src.config import N_MELS, N_MFCC, TARGET_SR
from src.models.inference import extract_legacy_features


@pytest.fixture
def synthetic_audio():
    """Create a 3-second synthetic audio signal with harmonic tones."""
    sr = TARGET_SR
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Mixture of fundamental frequency (150Hz) and upper harmonics (300Hz, 600Hz)
    signal = 0.4 * np.sin(2 * np.pi * 150 * t) + 0.2 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.sin(2 * np.pi * 600 * t)
    return signal.astype(np.float32), sr


def test_handcrafted_feature_names_count():
    """Verify that get_handcrafted_feature_names returns exactly 300 named dimensions."""
    names = get_handcrafted_feature_names()
    assert len(names) == 300
    assert "mfcc_1_mean" in names
    assert "mfcc_1_std" in names
    assert "rms_mean" in names
    assert "spectral_centroid_mean" in names
    assert "chroma_C_mean" in names
    assert "tonnetz_dim1_mean" in names


def test_extract_handcrafted_features_all(synthetic_audio):
    """Verify full handcrafted feature extraction returns deterministic (300,) float32 array."""
    y, sr = synthetic_audio
    features1 = extract_handcrafted_features(y, sr=sr, subset="all")
    features2 = extract_handcrafted_features(y, sr=sr, subset="all")

    assert features1.shape == (300,)
    assert features1.dtype == np.float32
    assert not np.isnan(features1).any(), "Extracted features contain NaN values"
    assert not np.isinf(features1).any(), "Extracted features contain Inf values"
    # Determinism
    assert np.allclose(features1, features2)


def test_extract_handcrafted_features_subsets(synthetic_audio):
    """Verify feature ablation subsets return exact expected dimensions."""
    y, sr = synthetic_audio
    feat_mfcc = extract_handcrafted_features(y, sr=sr, subset="mfcc_only")
    feat_spec = extract_handcrafted_features(y, sr=sr, subset="mfcc_spectral")

    assert feat_mfcc.shape == (240,)
    assert feat_spec.shape == (264,)


def test_extract_mel_spectrogram_shape(synthetic_audio):
    """Verify Mel-spectrogram output shape (128, 130) and normalized [-1, 1] range."""
    y, sr = synthetic_audio
    mel = extract_mel_spectrogram(y, sr=sr)

    assert mel.shape == (128, 130)
    assert mel.dtype == np.float32
    assert mel.min() >= -1.01
    assert mel.max() <= 1.01


def test_legacy_feature_extraction(synthetic_audio):
    """Verify legacy feature extractor outputs exactly 194 dimensions matching hybrid model."""
    y, sr = synthetic_audio
    legacy_feat = extract_legacy_features(y, sr=sr)

    assert legacy_feat.shape == (194,)
    assert legacy_feat.dtype == np.float32
    assert not np.isnan(legacy_feat).any()
