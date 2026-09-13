"""2D Convolutional Neural Network (CNN) for Mel-Spectrogram Speech Emotion Recognition.

Processes 2D time-frequency representations (Log-Mel Spectrograms) using
convolutional feature maps, batch normalization, spatial pooling, and global
average pooling, mapping time-frequency audio patterns to emotion probabilities.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    GroupNormalization,
    Input,
    MaxPooling2D,
    ReLU,
)
from tensorflow.keras.models import Model, load_model

from src.config import CNN_MODEL_PATH, RANDOM_SEED

logger = logging.getLogger(__name__)


def build_mel_cnn(
    input_shape: Tuple[int, int, int] = (128, 130, 1),
    num_classes: int = 8,
    dropout_rate: float = 0.35
) -> Model:
    """Build a lightweight, performant 2D CNN for Mel-spectrogram classification.

    Uses GroupNormalization instead of BatchNormalization to eliminate batch-size
    instabilities and prevent running-mean covariate drift across speakers.
    """
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    inputs = Input(shape=input_shape, name="mel_spectrogram_input")

    # Block 1: Low-level acoustic edges and formant onsets
    x = Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    x = GroupNormalization(groups=4, name="gn1")(x)
    x = ReLU(name="relu1")(x)
    x = MaxPooling2D((2, 2), name="pool1")(x)
    x = Dropout(0.2, name="drop1")(x)

    # Block 2: Intermediate harmonic structures and pitch contours
    x = Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = GroupNormalization(groups=8, name="gn2")(x)
    x = ReLU(name="relu2")(x)
    x = MaxPooling2D((2, 2), name="pool2")(x)
    x = Dropout(0.25, name="drop2")(x)

    # Block 3: High-level emotional prosodic descriptors
    x = Conv2D(128, (3, 3), padding="same", name="last_conv_layer")(x)
    x = GroupNormalization(groups=8, name="gn3")(x)
    x = ReLU(name="relu3")(x)

    # Global spatial aggregation across frequency and time
    x = GlobalAveragePooling2D(name="gap")(x)

    # Classification head
    x = Dense(64, activation="relu", name="dense_fc")(x)
    x = Dropout(dropout_rate, name="drop_fc")(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="MelSpectrogramCNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_mel_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    save_path: Path = CNN_MODEL_PATH,
    epochs: int = 35,
    batch_size: int = 32
) -> Tuple[Model, tf.keras.callbacks.History]:
    """Train the Mel-Spectrogram CNN with early stopping and learning rate reduction."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure channel dimension: (N, 128, 130, 1)
    if X_train.ndim == 3:
        X_train = np.expand_dims(X_train, axis=-1)
    if X_val.ndim == 3:
        X_val = np.expand_dims(X_val, axis=-1)

    model = build_mel_cnn(input_shape=X_train.shape[1:], num_classes=len(np.unique(y_train)))

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(save_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        )
    ]

    logger.info(f"Training Mel-Spectrogram CNN ({X_train.shape[0]} train, {X_val.shape[0]} val samples)...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Save final best model
    model.save(str(save_path))
    logger.info(f"Saved best Mel-Spectrogram CNN to {save_path}")
    return model, history
