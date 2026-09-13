"""AutoOptimizeML Adapter definition for EmotionAnalyzer (Speech Emotion Recognition model)."""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

# Custom Layers defined in EmotionAnalyzer
class TimeDistributedSum(Layer):
    def call(self, inputs):
        return K.sum(inputs, axis=1)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

class ExpandDims(Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)
    def call(self, inputs):
        return K.expand_dims(inputs, axis=self.axis)
    def compute_output_shape(self, input_shape):
        return input_shape + (1,)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hybrid_ser_model.h5")

def get_model():
    """Load the trained EmotionAnalyzer Keras Hybrid CNN+BiLSTM+Attention model."""
    with custom_object_scope({"ExpandDims": ExpandDims, "TimeDistributedSum": TimeDistributedSum}):
        model = load_model(MODEL_PATH)
    return model

def get_sample_input(batch_size: int = 1):
    """Sample audio feature tensor: shape (batch_size, 194, 1)."""
    np.random.seed(42)
    return np.random.randn(batch_size, 194, 1).astype(np.float32)

def get_test_data(n_samples: int = 50):
    """Test evaluation dataset: 194 features, 8 emotion classes."""
    np.random.seed(42)
    X = np.random.randn(n_samples, 194, 1).astype(np.float32)
    y = np.random.randint(0, 8, size=(n_samples,)).astype(np.int64)
    return X, y
