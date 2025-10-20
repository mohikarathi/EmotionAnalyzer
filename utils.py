import librosa
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.layers import Layer
import joblib

# ---------------- Custom Layers ----------------
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

# ---------------- Feature Extraction ----------------
def extract_audio_features(path):
    y_audio, sr = librosa.load(path, sr=None)
    y_audio = librosa.util.normalize(y_audio.astype(np.float32))

    mfcc = np.mean(librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=40).T, axis=0)
    mel = np.mean(librosa.power_to_db(librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128)).T, axis=0)
    spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y_audio, sr=sr).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y_audio, sr=sr).T, axis=0)
    y_harmonic = librosa.effects.harmonic(y_audio)
    tonnetz = np.mean(librosa.feature.tonnetz(y=y_harmonic, sr=sr).T, axis=0)
    y_harmonic, _ = librosa.effects.hpss(y_audio)
    poly = np.mean(y_harmonic)

    feature_vector = np.hstack([mfcc, mel, spec_contrast, chroma, tonnetz, poly])
    return feature_vector

# ---------------- Load Model & Encoder ----------------
MODEL_PATH = "models/hybrid_ser_model.h5"
ENCODER_PATH = "models/label_encoder.pkl"

with custom_object_scope({"ExpandDims": ExpandDims, "TimeDistributedSum": TimeDistributedSum}):
    model = load_model(MODEL_PATH)

encoder = joblib.load(ENCODER_PATH)

# ---------------- Prediction ----------------
def predict_emotion(audio_path):
    features = extract_audio_features(audio_path)
    features = np.expand_dims(features, axis=(0, -1))  # shape (1, 194, 1)
    preds = model.predict(features)
    emotion = encoder.inverse_transform([np.argmax(preds)])
    return emotion[0]
