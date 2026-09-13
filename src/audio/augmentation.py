"""Lightweight audio data augmentation for Speech Emotion Recognition training.

CRITICAL: Augmentations are applied STRICTLY during training.
They must NEVER be applied to validation or test evaluations.

Techniques implemented:
1. Additive Gaussian noise (simulates background recording environment)
2. Random gain / volume shifts (simulates speaker distance variance)
3. Pitch shifting (simulates vocal tract fundamental frequency variations)
4. Time stretching (simulates speaking rate / cadence variations)
"""

import logging
from typing import Optional

import librosa
import numpy as np

from src.config import RANDOM_SEED, TARGET_SR

logger = logging.getLogger(__name__)


def add_gaussian_noise(
    y: np.ndarray,
    min_snr_db: float = 15.0,
    max_snr_db: float = 30.0,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """Add subtle Gaussian white noise at a controlled Signal-to-Noise Ratio (SNR).

    Args:
        y: 1D audio waveform.
        min_snr_db: Minimum SNR in decibels (noisier).
        max_snr_db: Maximum SNR in decibels (cleaner).
        rng: Random state for reproducibility.

    Returns:
        Augmented 1D audio array.
    """
    if rng is None:
        rng = np.random.RandomState()

    signal_power = np.mean(y ** 2)
    if signal_power < 1e-8:
        return y

    target_snr = rng.uniform(min_snr_db, max_snr_db)
    noise_power = signal_power / (10 ** (target_snr / 10.0))
    noise = rng.normal(0, np.sqrt(noise_power), size=len(y)).astype(np.float32)

    return (y + noise).astype(np.float32)


def shift_pitch(
    y: np.ndarray,
    sr: int = TARGET_SR,
    n_steps: float = 1.0
) -> np.ndarray:
    """Shift pitch up or down by fractional semitones without altering duration.

    Args:
        y: 1D audio waveform.
        sr: Sampling rate.
        n_steps: Semitone offset (e.g. -1.5 to +1.5).

    Returns:
        Pitch-shifted 1D audio array.
    """
    try:
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps).astype(np.float32)
    except Exception as e:
        logger.debug(f"Pitch shift skipped: {e}")
        return y


def stretch_time(y: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """Stretch time (tempo change) without altering pitch.

    Args:
        y: 1D audio waveform.
        rate: Speed factor (e.g. 0.9 for slower, 1.1 for faster).

    Returns:
        Time-stretched 1D audio array.
    """
    if np.isclose(rate, 1.0):
        return y
    try:
        stretched = librosa.effects.time_stretch(y=y, rate=rate)
        return stretched.astype(np.float32)
    except Exception as e:
        logger.debug(f"Time stretch skipped: {e}")
        return y


def apply_gain(
    y: np.ndarray,
    min_gain_db: float = -4.0,
    max_gain_db: float = 4.0,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """Apply slight random gain/attenuation in decibels.

    Args:
        y: 1D audio waveform.
        min_gain_db: Minimum gain in dB.
        max_gain_db: Maximum gain in dB.
        rng: Random state.

    Returns:
        Gain-adjusted 1D audio array.
    """
    if rng is None:
        rng = np.random.RandomState()
    gain_db = rng.uniform(min_gain_db, max_gain_db)
    factor = 10.0 ** (gain_db / 20.0)
    return (y * factor).astype(np.float32)


def augment_audio(
    y: np.ndarray,
    sr: int = TARGET_SR,
    enable_noise: bool = True,
    enable_gain: bool = True,
    enable_pitch: bool = True,
    enable_stretch: bool = False,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """Apply a randomized, lightweight augmentation pipeline for training samples.

    Args:
        y: 1D audio waveform.
        sr: Sampling rate.
        enable_noise: If True, add slight Gaussian noise with 50% probability.
        enable_gain: If True, apply gain variation with 50% probability.
        enable_pitch: If True, shift pitch slightly with 50% probability.
        enable_stretch: If True, stretch time slightly with 50% probability.
        rng: Random state.

    Returns:
        Augmented 1D audio array.
    """
    if rng is None:
        rng = np.random.RandomState()

    y_aug = y.copy()

    # 1. Random Gain adjustment
    if enable_gain and rng.rand() > 0.5:
        y_aug = apply_gain(y_aug, rng=rng)

    # 2. Additive Noise
    if enable_noise and rng.rand() > 0.5:
        y_aug = add_gaussian_noise(y_aug, rng=rng)

    # 3. Pitch Shift (+/- 1.5 semitones)
    if enable_pitch and rng.rand() > 0.5:
        steps = rng.uniform(-1.5, 1.5)
        y_aug = shift_pitch(y_aug, sr=sr, n_steps=steps)

    # 4. Time Stretch (0.95 to 1.05)
    if enable_stretch and rng.rand() > 0.5:
        rate = rng.uniform(0.95, 1.05)
        y_aug = stretch_time(y_aug, rate=rate)

    # Normalize amplitude after augmentation
    max_amp = np.max(np.abs(y_aug))
    if max_amp > 1e-6:
        y_aug = y_aug / max_amp

    return y_aug.astype(np.float32)
