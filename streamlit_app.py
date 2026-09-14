"""EmotionAnalyzer — Streamlit Web Interface for Speech Emotion Recognition.

An explainable, multi-model portfolio application for acoustic speech emotion recognition.
Uses the modular 'src' inference engine, preprocessing pipeline, and explainability routines.
"""

import io
import os
from pathlib import Path
import tempfile
from typing import Optional

import numpy as np
import streamlit as st

from src.audio.preprocessing import preprocess_audio
from src.audio.visualization import (
    analyze_spectrogram_cues,
    compute_waveform_points,
    generate_melspectrogram_base64,
)
from src.config import CONFIDENCE_THRESHOLD, EMOTION_CLASSES, TARGET_SR
from src.models.explainability import explain_prediction
from src.models.inference import get_inference_engine

# Emotion display colors
EMOTION_META = {
    "angry": {"color": "#EF4444"},
    "calm": {"color": "#06B6D4"},
    "disgust": {"color": "#10B981"},
    "fearful": {"color": "#8B5CF6"},
    "happy": {"color": "#F59E0B"},
    "neutral": {"color": "#6B7280"},
    "sad": {"color": "#3B82F6"},
    "surprised": {"color": "#EC4899"},
}

# Streamlit Page Config
st.set_page_config(
    page_title="EmotionAnalyzer — Explainable SER",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
  .metric-card {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
  }
  .cue-title {
    font-weight: 600;
    font-size: 0.95rem;
    color: #F8FAFC;
    margin-bottom: 4px;
  }
  .cue-obs {
    font-size: 0.82rem;
    font-family: monospace;
    color: #818CF8;
    margin-bottom: 6px;
  }
  .cue-desc {
    font-size: 0.85rem;
    color: #94A3B8;
    line-height: 1.4;
  }
  .guide-pill {
    background: #0F172A;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
  }
  .guide-pill-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #E2E8F0;
  }
  .guide-pill-text {
    font-size: 0.78rem;
    color: #94A3B8;
    margin: 0;
  }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Pre-warming ML models and feature extraction engine...")
def load_cached_engine(version: str = "v2.2"):
    """Load and warm up model weights once per server lifetime."""
    engine = get_inference_engine()
    engine.warmup()
    return engine


engine = load_cached_engine("v2.2")

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("EmotionAnalyzer")
    st.caption("Explainable Speech Emotion Recognition System")

    st.markdown("---")
    st.subheader("Model Selection")
    model_choice = st.selectbox(
        "Active Classifier",
        options=["hybrid", "cnn", "random_forest"],
        format_func=lambda x: {
            "hybrid": "Hybrid SER Model (BiLSTM + Attention)",
            "cnn": "Mel-Spectrogram 2D CNN",
            "random_forest": "Random Forest Baseline (300-d Handcrafted)",
        }.get(x, x),
        help="Select the underlying machine learning model used to predict the emotion."
    )

    threshold = st.slider(
        "Confidence Ambiguity Threshold",
        min_value=0.10,
        max_value=0.70,
        value=CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Predictions below this probability are flagged as ambiguous."
    )

    st.markdown("---")
    st.subheader("Held-Out Speaker Benchmark")
    st.caption("Evaluated on unseen test speakers (Actors 21–24):")
    st.dataframe({
        "Model": ["Random Forest", "Logistic Reg.", "Mel CNN"],
        "Accuracy": ["47.92%", "46.25%", "17.92%"],
        "Macro F1": ["0.4445", "0.4652", "0.0780"],
        "Latency": ["24.8 ms", "0.13 ms", "46.0 ms"]
    }, hide_index=True)

    st.markdown("---")
    st.markdown("[View Source on GitHub](https://github.com/mohikarathi/EmotionAnalyzer)")


# ----------------- MAIN UI -----------------
st.title("Explainable Speech Emotion Recognition")
st.markdown(
    "Analyze vocal prosody, spectral energy, and emotional state from audio recordings. "
    "Features probability distributions, Log-Mel spectrogram visualization, and acoustic explainability."
)

tab_record, tab_upload = st.tabs(["Microphone Record", "Upload Audio File"])

audio_bytes: Optional[bytes] = None
audio_source_name = "recording.wav"

with tab_record:
    st.write("Record speech using your browser's microphone:")
    recorded_file = st.audio_input("Click the microphone to start recording")
    if recorded_file is not None:
        audio_bytes = recorded_file.getvalue()
        audio_source_name = "microphone_recording.wav"

with tab_upload:
    st.write("Upload a pre-recorded WAV, MP3, FLAC, or OGG file:")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "flac", "ogg", "m4a", "webm"],
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        audio_bytes = uploaded_file.getvalue()
        audio_source_name = uploaded_file.name

if audio_bytes is not None:
    st.audio(audio_bytes)

    if st.button("Analyze Speech Emotion", type="primary", use_container_width=True):
        with st.spinner("Processing audio and extracting acoustic features..."):
            temp_path = None
            try:
                # 1. Write audio bytes to temporary file with correct extension for robust format decoding
                ext = Path(audio_source_name).suffix.lower()
                if not ext:
                    ext = ".wav"
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(audio_bytes)
                    temp_path = tmp.name

                # 2. Run model inference directly on raw audio file to preserve natural, unpadded audio
                pred_result = engine.predict(temp_path, model_name=model_choice, threshold=threshold)
                pred_emotion = pred_result["predicted_emotion"]
                confidence = pred_result["confidence"]
                is_low_conf = pred_result["is_low_confidence"]
                probabilities = pred_result["probabilities"]

                # 3. Preprocess audio strictly for Mel-spectrogram visualization & acoustic cues
                y, sr = preprocess_audio(temp_path)
                melspec_b64 = generate_melspectrogram_base64(y, sr=sr)
                spectrogram_cues = analyze_spectrogram_cues(y, sr=sr, predicted_emotion=pred_emotion)

                # 4. Model explainability
                explanation = None
                try:
                    explanation = explain_prediction(temp_path, model_name=model_choice)
                except Exception as expl_err:
                    st.warning(f"Note: Explainability skipped: {expl_err}")

                st.markdown("---")

                # Layout: Left = Prediction & Probabilities, Right = Spectrogram & Cues
                col_left, col_right = st.columns([1, 1.2], gap="large")

                with col_left:
                    st.subheader("Inference Result")
                    meta = EMOTION_META.get(pred_emotion, {"color": "#6366F1"})

                    st.markdown(f"""
                    <div style="background-color: #1E293B; border-left: 5px solid {meta['color']}; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                      <span style="font-size: 0.85rem; text-transform: uppercase; color: #94A3B8; letter-spacing: 1px;">Predicted Acoustic Emotion</span>
                      <h1 style="margin: 6px 0; color: #FFFFFF; font-size: 2.2rem;">{pred_emotion.capitalize()}</h1>
                      <div style="display: flex; gap: 12px; align-items: center; margin-top: 8px;">
                        <span style="font-size: 1.1rem; font-weight: 700; color: {meta['color']};">{confidence*100:.1f}% Confidence</span>
                        <span style="background-color: {'#B45309' if is_low_conf else '#065F46'}; color: #FFFFFF; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">
                          {'Low Confidence / Ambiguous' if is_low_conf else 'High Confidence'}
                        </span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if is_low_conf:
                        st.warning(
                            f"**Acoustic Ambiguity Notice:** The highest confidence score ({confidence*100:.1f}%) "
                            f"is below the uncertainty threshold ({threshold*100:.0f}%). The clip may contain subtle or overlapping emotional prosody."
                        )

                    st.markdown("#### Probability Distribution")
                    for item in probabilities:
                        e_name = item["emotion"].capitalize()
                        e_score = item["score"]
                        e_pct = item["percentage"]
                        col_e, col_bar = st.columns([1, 3])
                        col_e.write(f"**{e_name}**")
                        col_bar.progress(float(e_score), text=f"{e_pct}%")

                with col_right:
                    st.subheader("Time-Frequency Representation")
                    st.caption("Log-scaled Mel-spectrogram computed across 128 filter banks (0 – 8 kHz)")

                    st.image(melspec_b64, use_container_width=True)

                    # Visual Guide Pills
                    st.markdown("""
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 16px;">
                      <div class="guide-pill">
                        <div class="guide-pill-title">Time (X-Axis)</div>
                        <p class="guide-pill-text">0 – 3.0s duration. Bright segments show active speech.</p>
                      </div>
                      <div class="guide-pill">
                        <div class="guide-pill-title">Frequency (Y-Axis)</div>
                        <p class="guide-pill-text">0 – 8 kHz Mel scale. &lt;1 kHz = pitch, &gt;2.5 kHz = friction.</p>
                      </div>
                      <div class="guide-pill">
                        <div class="guide-pill-title">Intensity (dB)</div>
                        <p class="guide-pill-text">Yellow/orange = peak energy; dark purple = silence.</p>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### Acoustic Cues Influencing Classification")
                    st.caption(spectrogram_cues.get("summary", "Key acoustic dimensions visible in this clip:"))

                    for cue in spectrogram_cues.get("cues", []):
                        st.markdown(f"""
                        <div class="metric-card">
                          <div class="cue-title">{cue['dimension']}</div>
                          <div class="cue-obs">{cue['observation']}</div>
                          <div class="cue-desc">{cue['interpretation']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # ----------------- EXPLAINABILITY -----------------
                if explanation is not None:
                    st.markdown("---")
                    st.subheader("Model Attribution & Explainability")
                    st.caption(explanation.get("description", "Acoustic feature attributions driving this prediction:"))

                    if explanation.get("type") == "feature_importance":
                        top_features = explanation.get("top_features", [])
                        cols = st.columns(min(3, len(top_features)))
                        for idx, feat in enumerate(top_features):
                            with cols[idx % len(cols)]:
                                st.markdown(f"""
                                <div class="metric-card">
                                  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                    <code style="font-size: 0.8rem; color: #818CF8;">{feat['name']}</code>
                                    <span style="font-weight: 700; font-size: 0.85rem; color: #34D399;">{feat['importance']*100:.2f}%</span>
                                  </div>
                                  <div style="font-size: 0.8rem; color: #94A3B8;">{feat['interpretation']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    elif explanation.get("type") == "gradcam" and explanation.get("heatmap_b64"):
                        st.image(explanation["heatmap_b64"], caption="Grad-CAM Saliency: Acoustic Attribution Regions", use_container_width=True)

            except Exception as exc:
                st.error(f"Error analyzing audio: {exc}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
else:
    st.info("Record your voice or upload an audio file above to begin analysis.")
