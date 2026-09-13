"""Central configuration module for EmotionAnalyzer.

Defines all constants, hyperparameters, paths, and application settings
to ensure reproducibility and avoid hardcoded values across the codebase.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = Path("/home/hp/Downloads/DL/Project/RAVDESS")  # Local RAVDESS dataset
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
UPLOADS_DIR = PROJECT_ROOT / "uploads"

# Audio Preprocessing Parameters
TARGET_SR = 22050          # Standard sampling rate for speech processing (Hz)
AUDIO_DURATION = 3.0       # Standardized clip duration (seconds)
TARGET_SAMPLES = int(TARGET_SR * AUDIO_DURATION)  # 66,150 samples
TOP_DB_TRIM = 30           # Threshold (in dB) below peak for silence trimming
N_FFT = 2048               # FFT window size
HOP_LENGTH = 512           # Hop length for STFT (~23ms at 22050Hz)
N_MELS = 128               # Number of Mel filter banks
N_MFCC = 40                # Number of MFCC coefficients

# Model Paths
LEGACY_MODEL_PATH = MODELS_DIR / "hybrid_ser_model.h5"
LEGACY_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
CNN_MODEL_PATH = MODELS_DIR / "cnn_mel_model.keras"
RF_MODEL_PATH = MODELS_DIR / "classical_rf.joblib"
LOGREG_MODEL_PATH = MODELS_DIR / "classical_logreg.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
FEATURE_CONFIG_PATH = MODELS_DIR / "feature_config.json"

# Inference & Uncertainty
CONFIDENCE_THRESHOLD = 0.35  # Threshold below which prediction is marked as low confidence

# Emotion Label Mapping (standard 8 RAVDESS emotions)
EMOTION_CLASSES = [
    "angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"
]

RAVDESS_CODE_TO_EMOTION = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Speaker-Independent Split (Actors 01-24)
# 18 speakers train (75%), 2 speakers val (8.3%), 4 speakers test (16.7%)
# Both genders evenly represented: odd = male, even = female
TRAIN_SPEAKERS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
]
VAL_SPEAKERS = [19, 20]
TEST_SPEAKERS = [21, 22, 23, 24]

# Web & Security Settings
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 10000))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "t")

# Reproducibility
RANDOM_SEED = 42
