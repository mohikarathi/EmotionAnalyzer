"""Classical Machine Learning Baseline Pipelines for Speech Emotion Recognition.

Implements Scikit-learn pipelines with StandardScaler and classifiers:
- Random Forest Classifier
- Multinomial Logistic Regression
- Support Vector Classifier (RBF Kernel with probability calibration)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import MODELS_DIR, RANDOM_SEED
from src.evaluation.metrics import compute_ser_metrics

logger = logging.getLogger(__name__)


def build_classical_pipeline(model_type: str = "random_forest") -> Pipeline:
    """Construct an end-to-end scikit-learn pipeline with feature scaling.

    Args:
        model_type: Classifier architecture ('random_forest', 'logistic_regression', 'svc').

    Returns:
        Configured scikit-learn Pipeline instance.
    """
    if model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=16,
            min_samples_split=4,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight="balanced"
        )
    elif model_type == "logistic_regression":
        clf = LogisticRegression(
            max_iter=1000,
            C=0.5,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            solver="lbfgs"
        )
    elif model_type == "svc":
        clf = SVC(
            kernel="rbf",
            C=1.5,
            gamma="scale",
            probability=True,
            random_state=RANDOM_SEED,
            class_weight="balanced"
        )
    else:
        raise ValueError(f"Unknown classical model type: {model_type}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf)
    ])
    return pipeline


def train_classical_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "random_forest"
) -> Pipeline:
    """Train a classical machine learning pipeline on feature matrices.

    Args:
        X_train: 2D feature matrix of shape (n_samples, n_features).
        y_train: 1D class labels array of shape (n_samples,).
        model_type: Classifier type.

    Returns:
        Fitted Pipeline.
    """
    logger.info(f"Training {model_type} on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
    pipeline = build_classical_pipeline(model_type)
    pipeline.fit(X_train, y_train)
    logger.info(f"Trained {model_type} pipeline successfully.")
    return pipeline


def save_classical_model(pipeline: Pipeline, save_path: Path):
    """Serialize trained classical pipeline to disk."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    logger.info(f"Saved classical model pipeline to {save_path}")


def load_classical_model(model_path: Path) -> Pipeline:
    """Load serialized classical pipeline from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)
