"""Comprehensive feature extraction module for Speech Emotion Recognition.

Implements deterministic acoustic feature extraction pipelines:
1. Handcrafted feature vectors (MFCCs, deltas, delta-deltas, spectral descriptors,
   chroma, tonnetz, RMS, ZCR) with mean + std statistical temporal aggregation.
2. 2D Log-Mel Spectrogram extraction for Convolutional Neural Networks.
3. Legacy 194-dimensional feature extraction for backward compatibility with
   the original hybrid SER model.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np

from src.audio.preprocessing import preprocess_audio
from src.config import (
    FEATURE_CONFIG_PATH,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    N_MFCC,
    TARGET_SR,
)


def get_handcrafted_feature_names(n_mfcc: int = N_MFCC) -> List[str]:
    """Return explicit human-readable names for all handcrafted feature dimensions.

    Enables transparent feature-importance interpretation and error attribution.

    Returns:
        List of feature name strings corresponding 1-to-1 with extracted feature array.
    """
    names = []

    # MFCCs (mean and std)
    for i in range(1, n_mfcc + 1):
        names.append(f"mfcc_{i}_mean")
        names.append(f"mfcc_{i}_std")

    # MFCC Deltas (mean and std)
    for i in range(1, n_mfcc + 1):
        names.append(f"mfcc_delta_{i}_mean")
        names.append(f"mfcc_delta_{i}_std")

    # MFCC Delta-Deltas (mean and std)
    for i in range(1, n_mfcc + 1):
        names.append(f"mfcc_delta2_{i}_mean")
        names.append(f"mfcc_delta2_{i}_std")

    # Spectral descriptors
    names.extend(["zcr_mean", "zcr_std"])
    names.extend(["rms_mean", "rms_std"])
    names.extend(["spectral_centroid_mean", "spectral_centroid_std"])
    names.extend(["spectral_bandwidth_mean", "spectral_bandwidth_std"])
    names.extend(["spectral_rolloff_mean", "spectral_rolloff_std"])

    # Spectral contrast (7 bands)
    for i in range(1, 8):
        names.append(f"spectral_contrast_b{i}_mean")
        names.append(f"spectral_contrast_b{i}_std")

    # Chroma STFT (12 pitch classes)
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for pc in pitch_classes:
        names.append(f"chroma_{pc}_mean")
        names.append(f"chroma_{pc}_std")

    # Tonnetz (6 tonal centroid dimensions)
    for i in range(1, 7):
        names.append(f"tonnetz_dim{i}_mean")
        names.append(f"tonnetz_dim{i}_std")

    return names


def extract_handcrafted_features(
    audio_path_or_y: Union[str, Path, np.ndarray],
    sr: int = TARGET_SR,
    subset: str = "all"
) -> np.ndarray:
    """Extract deterministic 1D handcrafted acoustic feature vector.

    Aggregation Strategy:
    Audio frame-level time series are summarized by computing the empirical Mean
    and Standard Deviation over time.
    - Mean captures the global central spectral tendency (average formant envelope,
      overall brightness, average energy).
    - Standard Deviation captures emotional prosodic dynamics (pitch inflection,
      vocal jitter, emotional energy variation).

    Subsets supported for ablation experiments:
    - 'mfcc_only': 40 MFCCs + deltas + delta2 (240 dims)
    - 'mfcc_spectral': MFCCs + ZCR, RMS, Centroid, Bandwidth, Rolloff, Contrast (260 dims)
    - 'all': Full set (+ Chroma, Tonnetz) (300 dims)

    Args:
        audio_path_or_y: Audio filepath or 1D float32 audio waveform.
        sr: Sampling rate.
        subset: Feature ablation subset ('all', 'mfcc_only', 'mfcc_spectral').

    Returns:
        1D float32 array of deterministic length.
    """
    if isinstance(audio_path_or_y, (str, Path)) or hasattr(audio_path_or_y, "read"):
        y, sr = preprocess_audio(audio_path_or_y, target_sr=sr)
    else:
        y = audio_path_or_y

    features = []

    # 1. MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    for coeff in mfcc:
        features.extend([float(np.mean(coeff)), float(np.std(coeff))])

    # 2. MFCC Deltas (velocity)
    mfcc_delta = librosa.feature.delta(mfcc, order=1)
    for coeff in mfcc_delta:
        features.extend([float(np.mean(coeff)), float(np.std(coeff))])

    # 3. MFCC Delta-Deltas (acceleration)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    for coeff in mfcc_delta2:
        features.extend([float(np.mean(coeff)), float(np.std(coeff))])

    if subset == "mfcc_only":
        return np.array(features, dtype=np.float32)

    # 4. Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)[0]
    features.extend([float(np.mean(zcr)), float(np.std(zcr))])

    # 5. RMS Energy
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    features.extend([float(np.mean(rms)), float(np.std(rms))])

    # 6. Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    features.extend([float(np.mean(spec_cent)), float(np.std(spec_cent))])

    # 7. Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    features.extend([float(np.mean(spec_bw)), float(np.std(spec_bw))])

    # 8. Spectral Rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    features.extend([float(np.mean(spec_rolloff)), float(np.std(spec_rolloff))])

    # 9. Spectral Contrast (7 bands)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    for band in spec_contrast:
        features.extend([float(np.mean(band)), float(np.std(band))])

    if subset == "mfcc_spectral":
        return np.array(features, dtype=np.float32)

    # 10. Chroma STFT (12 pitch classes)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    for pitch in chroma:
        features.extend([float(np.mean(pitch)), float(np.std(pitch))])

    # 11. Tonnetz (tonal centroids)
    y_harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    for dim in tonnetz:
        features.extend([float(np.mean(dim)), float(np.std(dim))])

    return np.array(features, dtype=np.float32)


def extract_mel_spectrogram(
    audio_path_or_y: Union[str, Path, np.ndarray],
    sr: int = TARGET_SR,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: HOP_LENGTH = HOP_LENGTH
) -> np.ndarray:
    """Extract normalized Log-Mel Spectrogram array for 2D CNN input.

    Computes:
    STFT -> Mel Filter Bank -> Decibel Scaling -> Standardized [-1, 1] range.

    Args:
        audio_path_or_y: Filepath or 1D audio array.
        sr: Sampling rate.
        n_mels: Number of Mel frequency bands (height).
        n_fft: FFT window size.
        hop_length: Hop length between successive frames.

    Returns:
        2D float32 array of shape (n_mels, time_steps).
        For 3.0s audio at 22050Hz with hop_length=512, shape is (128, 130).
    """
    if isinstance(audio_path_or_y, (str, Path)) or hasattr(audio_path_or_y, "read"):
        y, sr = preprocess_audio(audio_path_or_y, target_sr=sr)
    else:
        y = audio_path_or_y

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=8000
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize from dB range (typically [-80, 0]) to [-1.0, 1.0]
    min_db = -80.0
    mel_norm = np.clip(mel_db, min_db, 0.0)
    mel_norm = 2.0 * ((mel_norm - min_db) / (-min_db)) - 1.0

    return mel_norm.astype(np.float32)


def save_feature_config(config_path: Path = FEATURE_CONFIG_PATH):
    """Persist feature configuration schema to JSON for strict training/inference parity."""
    feature_names = get_handcrafted_feature_names()
    config_dict = {
        "sampling_rate": TARGET_SR,
        "duration_seconds": 3.0,
        "n_mfcc": N_MFCC,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "handcrafted_dimension": len(feature_names),
        "feature_names": feature_names,
        "legacy_dimension": 194
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    return config_dict
