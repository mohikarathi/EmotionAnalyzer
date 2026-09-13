"""Audio preprocessing pipeline for Speech Emotion Recognition (SER).

This module defines deterministic, reusable preprocessing functions used
identically during training and inference.

Why each step exists:
1. Audio loading: Standardizes varied container formats (WAV, MP3, FLAC, OGG, WebM)
   into float32 numerical amplitude arrays.
2. Mono conversion: Emotion cues in vocal prosody (pitch, energy, timbre) do not
   depend on stereo spatial panning. Mono conversion reduces dimensionality and
   ensures channel invariance across recording equipment.
3. Sampling-rate normalization: Different microphones record at 16kHz, 44.1kHz,
   or 48kHz. Resampling to a unified sampling rate (default: 22050 Hz) ensures
   that the Nyquist frequency and discrete Fourier transform bin mappings
   are identical across all samples.
4. Silence trimming: Leading and trailing silence or low-level background noise
   contains zero acoustic emotion cues and dilutes global statistical aggregations.
   Trimming focuses the feature extractor on active speech.
5. Fixed-duration handling (pad / truncate): Neural networks (CNNs and BiLSTMs)
   require fixed-size input tensors. Uniform 3.0s duration aligns time-frequency
   spectrogram dimensions.
6. Amplitude normalization: Differences in microphone gain and speaking distance
   cause arbitrary volume variations. Normalizing peak amplitude to [-1.0, 1.0]
   prevents loudness bias from being mistaken for emotional intensity.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import io

import librosa
import numpy as np

from src.config import AUDIO_DURATION, TARGET_SAMPLES, TARGET_SR, TOP_DB_TRIM


def load_audio(
    file_path_or_buffer: Union[str, Path, io.BytesIO],
    target_sr: int = TARGET_SR,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """Load an audio file or in-memory byte buffer and convert to target sampling rate.

    Args:
        file_path_or_buffer: Path to audio file or file-like BytesIO stream.
        target_sr: Target sampling rate in Hz. If None, native rate is preserved.
        mono: If True, convert multi-channel audio to single-channel mono.

    Returns:
        Tuple of (audio_array, sampling_rate) where audio_array is np.float32.

    Raises:
        ValueError: If audio is empty or corrupt.
        RuntimeError: If librosa/soundfile cannot decode the file.
    """
    try:
        # librosa.load handles mono conversion and resampling via soxr/scipy
        y, sr = librosa.load(file_path_or_buffer, sr=target_sr, mono=mono)
    except Exception as exc:
        raise RuntimeError(f"Failed to load and decode audio: {exc}") from exc

    if y is None or len(y) == 0:
        raise ValueError("Decoded audio signal is empty.")

    y = y.astype(np.float32)
    return y, sr


def normalize_amplitude(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize audio amplitude to [-1.0, 1.0] range.

    Prevents gain differences and recording hardware variations from
    biasing feature representations.

    Args:
        y: 1D audio waveform array.
        eps: Small epsilon to prevent division by zero on silent clips.

    Returns:
        Normalized audio waveform array in float32.
    """
    max_amp = np.max(np.abs(y))
    if max_amp > eps:
        return y / max_amp
    return y


def trim_silence(
    y: np.ndarray,
    top_db: int = TOP_DB_TRIM,
    ref: float = np.max
) -> np.ndarray:
    """Remove leading and trailing silence below a decibel threshold.

    Focuses analysis on the phonated speech segments containing emotional prosody.

    Args:
        y: 1D audio waveform array.
        top_db: Decibels below reference peak considered silence (default: 30 dB).
        ref: Reference value for decibel calculation.

    Returns:
        Trimmed 1D audio waveform array.
    """
    if len(y) == 0 or np.max(np.abs(y)) < 1e-6:
        return y

    try:
        trimmed, _ = librosa.effects.trim(y, top_db=top_db, ref=ref)
        # Guard against over-trimming completely silent or near-silent files
        if len(trimmed) > 0:
            return trimmed
    except Exception:
        pass
    return y


def pad_or_trim_audio(
    y: np.ndarray,
    target_samples: int = TARGET_SAMPLES
) -> np.ndarray:
    """Ensure audio signal is exactly target_samples in length.

    If shorter, zeroes are symmetrically padded.
    If longer, centered truncation is applied to preserve core speech.

    Args:
        y: 1D audio waveform array.
        target_samples: Target length in discrete audio samples.

    Returns:
        1D audio waveform array of shape (target_samples,).
    """
    current_length = len(y)

    if current_length == target_samples:
        return y

    if current_length < target_samples:
        # Symmetric zero-padding
        pad_total = target_samples - current_length
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(y, (pad_left, pad_right), mode="constant", constant_values=0.0)

    # Center-crop longer audio
    excess = current_length - target_samples
    start = excess // 2
    return y[start : start + target_samples]


def preprocess_audio(
    file_path_or_buffer: Union[str, Path, io.BytesIO],
    target_sr: int = TARGET_SR,
    target_duration: float = AUDIO_DURATION,
    trim: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """Complete, end-to-end audio preprocessing pipeline.

    Applies the full sequence:
    1. Load & decode audio
    2. Resample to target_sr and convert to mono
    3. Peak amplitude normalization
    4. Optional silence trimming
    5. Duration normalization (symmetric pad or center-crop)
    6. Final amplitude safety check

    Args:
        file_path_or_buffer: Audio file path or byte stream.
        target_sr: Sampling rate (Hz).
        target_duration: Target duration in seconds.
        trim: Whether to trim leading/trailing silence.
        normalize: Whether to apply peak amplitude normalization.

    Returns:
        Tuple of (preprocessed_audio_array, target_sr).
        preprocessed_audio_array has exact shape (int(target_sr * target_duration),).
    """
    # 1 & 2: Load, resample, mono
    y, sr = load_audio(file_path_or_buffer, target_sr=target_sr, mono=True)

    # 3: Amplitude normalization
    if normalize:
        y = normalize_amplitude(y)

    # 4: Silence trimming
    if trim:
        y = trim_silence(y, top_db=TOP_DB_TRIM)

    # 5: Pad or trim to fixed duration
    target_samples = int(target_sr * target_duration)
    y = pad_or_trim_audio(y, target_samples=target_samples)

    # 6: Final normalization check
    if normalize:
        y = normalize_amplitude(y)

    return y.astype(np.float32), sr
