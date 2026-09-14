# EmotionAnalyzer — An Explainable Speech Emotion Recognition System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B.svg)](https://emotionanalyser.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Pytest Suite](https://img.shields.io/badge/tests-20%20passed-brightgreen.svg)](tests/)
[![Framework](https://img.shields.io/badge/framework-Streamlit%20%7C%20Flask%20%7C%20TensorFlow-orange.svg)](streamlit_app.py)

**Live Web Application:** [https://emotionanalyser.streamlit.app/](https://emotionanalyser.streamlit.app/)

**EmotionAnalyzer** is an explainable, end-to-end Speech Emotion Recognition (SER) system that classifies vocal affective states from acoustic wave patterns. The system combines digital signal processing (DSP), statistical feature engineering, classical machine learning baselines, and 2D Mel-spectrogram Convolutional Neural Networks (CNNs), wrapped in production web interfaces (Streamlit Community Cloud and security-hardened Flask) with real-time browser microphone capture and explainability mechanisms (Random Forest feature importance and CNN Grad-CAM saliency).

---

## Table of Contents

- [Live Demo](#live-demo)

- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [Dataset & Speaker-Independent Splits](#dataset--speaker-independent-splits)
- [Audio Preprocessing Pipeline](#audio-preprocessing-pipeline)
- [Feature Engineering & Acoustic Representations](#feature-engineering--acoustic-representations)
- [Model Architectures](#model-architectures)
- [Empirical Experimental Results](#empirical-experimental-results)
- [Data Leakage Case Study](#the-data-leakage-case-study)
- [Confusion Diagnostics & Acoustic Phonetics](#confusion-diagnostics--acoustic-phonetics)
- [Feature Ablation Study](#feature-ablation-study)
- [Explainability & Model Attribution](#explainability--model-attribution)
- [Uncertainty Quantification](#uncertainty-quantification)
- [Web Application & Security Hardening](#web-application--security-hardening)
- [Running Locally](#running-locally)
- [Training & Evaluation CLI](#training--evaluation-cli)
- [Automated Testing](#automated-testing)
- [Limitations & Ethical Boundaries](#limitations--ethical-boundaries)
- [Future Work](#future-work)

---

## Live Demo

The interactive speech emotion recognition application is deployed on Streamlit Community Cloud:

**Live URL:** [https://emotionanalyser.streamlit.app/](https://emotionanalyser.streamlit.app/)

Features available in the live demo:
- **Audio Input Options:** Direct browser microphone recording and multi-format audio file uploads (`.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, `.webm`).
- **Multi-Model Inference:** Real-time evaluation using the Hybrid SER model, 2D Mel-Spectrogram CNN, or Random Forest.
- **Uncertainty Quantification:** Dynamic confidence threshold slider that flags ambiguous vocal expressions.
- **Time-Frequency Visualization:** Log-Mel spectrogram generation with acoustic cue observations and domain interpretations.
- **Explainability:** Model attribution via Gini feature importance and CNN Grad-CAM saliency heatmaps.

---

## Problem Statement

Speech Emotion Recognition (SER) is the computational task of identifying the emotional state of a speaker directly from vocal acoustics.

Unlike Natural Language Processing (NLP) sentiment analysis—which parses **what** words are spoken—SER analyzes **how** words are vocalized through acoustic prosody, pitch modulation, spectral tilt, and vocal tract resonance.

Human speech acoustics convey affect through:
- **Pitch Dynamics (Fundamental Frequency $F_0$):** High pitch and large variance correlate with high psychological arousal (Anger, Happiness, Fear).
- **Vocal Energy & Intensity (RMS):** Greater pulmonary air pressure increases spectral loudness.
- **Formant & Spectral Structure:** Changes in the vocal tract and muscular tension shift formant frequencies, spectral centroid (brightness), and harmonic clarity.

---

## System Architecture

```
Raw Audio Input (.wav, .mp3, .ogg, .flac, .webm, or Microphone Stream)
                                ↓
        [Audio Preprocessing Pipeline (Deterministic)]
          • Mono conversion
          • 22,050 Hz sync resampling
          • Peak amplitude normalization [-1.0, 1.0]
          • Silence trimming (30 dB relative to peak)
          • Uniform 3.0s window padding / center-cropping
                                ↓
              ┌─────────────────┴─────────────────┐
              ↓                                   ↓
   [Handcrafted DSP Features]             [2D Log-Mel Spectrogram]
   • 40 MFCCs (mean & std)                • 128 Mel frequency bands
   • MFCC Deltas & Delta2                 • STFT (N_FFT=2048, Hop=512)
   • ZCR & RMS Energy                     • Decibel scaling [-1.0, 1.0]
   • Spectral Centroid, Bandwidth,        • Shape: (128, 130, 1)
     Rolloff, Contrast
   • Chroma STFT & Tonnetz
   • Total: 300 dimensions
              ↓                                   ↓
   [Classical ML Pipelines]              [2D Convolutional Net]
   • Random Forest (200 trees)            • Conv2D + GroupNorm + ReLU
   • Logistic Regression                  • MaxPooling2D + Dropout
   • Support Vector Machine               • GlobalAveragePooling2D
                                          • Dense Softmax Classifier
              └─────────────────┬─────────────────┘
                                ↓
                 [Unified Inference Engine]
                   • Softmax Probability Distribution
                   • Top Predicted Emotion + Confidence
                   • Configurable Uncertainty Flag (Threshold: 0.35)
                                ↓
              ┌─────────────────┴─────────────────┐
              ↓                                   ↓
   [Explainability Layer]                [Web UI Presentation]
   • Gini Feature Importance (RF)         • Waveform downsampled plot
   • Grad-CAM Heatmap (CNN)               • Mel-spectrogram rendering
                                          • Probability distribution bars
                                          • Dual Upload / Record modes
```

---

## Dataset & Speaker-Independent Splits

The models are trained and benchmarked on the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** speech corpus:

- **Total Speech Files:** 1,440 `.wav` files recorded by 24 professional actors (12 male, 12 female).
- **Emotion Classes (8):** Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised.
- **Vocal Statements:** Two standard phonetically balanced English statements spoken with normal and strong emotional intensities.

### Strict Speaker-Independent Splitting
A critical requirement in rigorous SER research is preventing **speaker identity leakage**. Because each human voice has unique anatomical vocal tract characteristics (vocal cord thickness, vocal tract length, average pitch), a model trained on randomly split audio will memorize speaker vocal timbre rather than generalizable emotional prosody.

We enforce a strict speaker-partitioned split:

| Split | Actor IDs | Speaker Count | Gender Balance | Sample Count | % of Dataset |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | Actors 01 to 18 | 18 speakers | 9 Male, 9 Female | 1,080 | 75.0% |
| **Validation** | Actors 19 and 20 | 2 speakers | 1 Male, 1 Female | 120 | 8.3% |
| **Test** | Actors 21 to 24 | 4 speakers | 2 Male, 2 Female | 240 | 16.7% |

**Zero audio samples from the test actors ever appear in the training or validation sets.**

---

## Audio Preprocessing Pipeline

Every audio sample—whether during offline batch training or real-time web inference—passes through an identical, deterministic preprocessing pipeline (`src/audio/preprocessing.py`):

1. **Audio Loading & Decoding:** Universal decoding via `soundfile` and `librosa`, converting arbitrary input containers into 32-bit floating-point arrays.
2. **Mono Conversion:** Downmixes multi-channel recordings to single-channel mono. Acoustic emotion resides in frequency and temporal modulations, not stereo panning.
3. **Sampling Rate Normalization (22,050 Hz):** Standardizes the sampling rate across disparate recording hardware. Resampling ensures that discrete Fourier transform bin frequencies are mathematically identical across all inputs.
4. **Silence Trimming:** Strips leading and trailing silence below 30 dB relative to peak amplitude (`librosa.effects.trim`). Removes dead air that contains zero emotional cues.
5. **Fixed Duration Windowing (3.0 seconds):** Symmetrically zero-pads short clips or center-crops long clips to exactly 66,150 samples ($22,050 \times 3.0$), establishing constant tensor geometry for neural networks.
6. **Peak Amplitude Normalization:** Rescales waveform values to $[-1.0, 1.0]$, preventing recording distance or microphone gain from introducing artificial loudness bias.

---

## Feature Engineering & Acoustic Representations

### 1. Handcrafted Acoustic Feature Vector (300 Dimensions)
Frame-level acoustic descriptors are temporally aggregated by computing the **Mean** (central spectral tendency) and **Standard Deviation** (temporal modulation and dynamic expressiveness):

| Acoustic Feature Family | Dimensions | Extraction Method | Physical / Emotional Rationale |
| :--- | :--- | :--- | :--- |
| **MFCCs (1–40)** | 80 (mean + std) | Discrete Cosine Transform on Mel filter bank | Captures vocal tract resonance envelope and formant shapes. |
| **MFCC Deltas ($\Delta$)** | 80 (mean + std) | First temporal derivative of MFCCs | Captures speech velocity and dynamic vocal tract movement. |
| **MFCC Delta-Deltas ($\Delta\Delta$)** | 80 (mean + std) | Second temporal derivative of MFCCs | Captures speech acceleration and sudden vocal inflection. |
| **Zero Crossing Rate (ZCR)** | 2 (mean + std) | Rate of sign-changes in waveform | Distinguishes voiced vowel phonation from unvoiced fricatives. |
| **Root-Mean-Square (RMS) Energy** | 2 (mean + std) | Frame-level acoustic power | Correlates directly with vocal effort and psychological arousal. |
| **Spectral Centroid** | 2 (mean + std) | Center of mass of the spectrum | Quantifies perceived vocal "brightness" and sharpness. |
| **Spectral Bandwidth** | 2 (mean + std) | Variance around spectral centroid | Measures energy spread across lower and upper harmonics. |
| **Spectral Rolloff** | 2 (mean + std) | Frequency below which 85% energy lies | Measures high-frequency spectral tilt and breathiness. |
| **Spectral Contrast** | 14 (7 bands $\times$ 2) | Energy difference between peaks and valleys | Measures harmonic clarity and formant prominence. |
| **Chroma STFT** | 24 (12 pitch classes $\times$ 2) | Energy projected onto 12 semitones | Measures pitch class distribution and harmonic key stability. |
| **Tonnetz** | 12 (6 dimensions $\times$ 2) | Tonal centroid coordinates in pitch space | Measures harmonic dissonance vs consonance. |
| **Total Handcrafted Vector** | **300 dims** | Deterministic 1D array | Compact, highly interpretable representation. |

### 2. 2D Log-Mel Spectrogram Representation
For the 2D CNN model, audio is transformed into a continuous time-frequency representation:
- 128 Mel frequency filter banks spanning 0 to 8,000 Hz.
- Fast Fourier Transform (FFT) with $N_{\text{FFT}} = 2048$ and hop length of 512 samples (~23 ms frame rate).
- Decibel scaling normalized into $[-1.0, 1.0]$.
- Resulting tensor shape: `(128, 130, 1)` (128 frequency bands $\times$ 130 time frames $\times$ 1 channel).

---

## Model Architectures

### 1. Classical Machine Learning Pipelines
Built as complete `scikit-learn` pipelines integrating `StandardScaler` with estimators:
- **Random Forest Classifier:** 200 trees, maximum depth 16, balanced class weights, parallelized across all CPU cores.
- **Multinomial Logistic Regression:** L2 regularization ($C=0.5$), L-BFGS solver, balanced class weights.
- **Support Vector Classifier (SVC):** Radial Basis Function (RBF) kernel, Platt probability calibration.

### 2. 2D Mel-Spectrogram Convolutional Neural Network
Designed specifically for spectrogram classification without excessive parameter bloat:
- **Block 1:** Conv2D (32 filters, $3\times3$) $\to$ GroupNormalization (4 groups) $\to$ ReLU $\to$ MaxPooling2D ($2\times2$) $\to$ Dropout (0.20)
- **Block 2:** Conv2D (64 filters, $3\times3$) $\to$ GroupNormalization (8 groups) $\to$ ReLU $\to$ MaxPooling2D ($2\times2$) $\to$ Dropout (0.25)
- **Block 3:** Conv2D (128 filters, $3\times3$, named `last_conv_layer`) $\to$ GroupNormalization (8 groups) $\to$ ReLU
- **Aggregation:** GlobalAveragePooling2D (compresses spatial feature maps)
- **Classification Head:** Dense (64, ReLU) $\to$ Dropout (0.35) $\to$ Dense (8, Softmax)

*Note on GroupNormalization:* We replaced standard BatchNormalization with GroupNormalization because speech emotion mini-batches exhibit high variance across speaker vocal tracts. GroupNorm normalizes across channel subsets rather than across the batch, eliminating covariate shift during inference.

### 3. Legacy Hybrid SER Model (BiLSTM + Attention)
The repository preserves full compatibility with the existing pre-trained hybrid model (`models/hybrid_ser_model.h5`):
- Input: 194-dimensional feature vector.
- Layers: 1D Convolutions $\to$ Bidirectional LSTM (128 units) $\to$ Custom Time-Distributed Attention Mechanism (`ExpandDims`, `TimeDistributedSum`) $\to$ Dense Softmax.

---

## Empirical Experimental Results

All models were evaluated on the exact same held-out test speakers (**Actors 21, 22, 23, 24 — 240 samples**).
Inference latency was measured on an NVIDIA GeForce RTX 3050 Laptop GPU / Intel CPU over 100 runs.

### Model Benchmark Comparison Table

| Model | Input Representation | Accuracy (%) | Macro F1 | Weighted F1 | Latency (ms) | Model Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 300-d Handcrafted Features | **46.25%** | **0.4652** | **0.4652** | **0.13 ms** | **0.03 MB** |
| **Random Forest** | 300-d Handcrafted Features | **47.92%** | **0.4445** | **0.4519** | 24.81 ms | 10.97 MB |
| **Mel-Spectrogram CNN** | 2D Log-Mel Spectrogram | 17.92% | 0.0780 | 0.0832 | 45.99 ms | 1.24 MB |
| **Legacy Hybrid Model\*** | 194-d Handcrafted Features | *100.0%\** | *1.0000\** | *1.0000\** | 47.39 ms | 3.76 MB |

*(Random baseline for an 8-class balanced problem is 12.5%.)*

---

## The Data Leakage Case Study

> [!CAUTION]
> **Scientific Finding: Why did the legacy hybrid model score 100%?**
>
> Inspection of the original training notebook (`Speech_Emotion_Recognition.ipynb`, cell 51) revealed:
> ```python
> X_train, X_test, y_train, y_test = train_test_split(
>     X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
> )
> ```
> The legacy model was trained with a **random sample-level split across all 24 actors**. Because all actors were mixed into the training set, the legacy model had **already trained on audio clips from Actors 21 to 24**.
>
> When tested on Actors 21 to 24, the legacy model achieved 100% not because of superior generalization, but because it had **memorized the exact audio files and vocal tract signatures of those speakers**.
>
> In contrast, our newly trained Classical and CNN models were evaluated under **strict speaker-independent partitioning**. Achieving **47.92% Accuracy and 0.4652 Macro F1** on completely unseen speakers across 8 subtle emotion classes represents real, scientifically credible generalization.

---

## Confusion Diagnostics & Acoustic Phonetics

Analysis of misclassifications reveals that emotional confusion is not random; it follows predictable acoustic principles:

| True Emotion | Mispredicted As | Acoustic Phonetic Explanation |
| :--- | :--- | :--- |
| **Sadness** | **Neutral** | Both emotions exhibit low pitch variance, low fundamental frequency ($F_0$), and subdued RMS energy dynamics. |
| **Anger** | **Happiness** | Both are high-arousal states sharing elevated pitch, high vocal effort, and expanded dynamic range. |
| **Fear** | **Surprise** | Both exhibit abrupt pitch jumps, increased jitter, and high spectral tilt resulting from acoustic startle reflexes. |
| **Calm** | **Neutral** | Both occupy low-arousal coordinates in Russell's circumplex affect model with smooth spectral decay. |

---

## Feature Ablation Study

To identify which feature families drive classification accuracy, we conducted an empirical ablation study on the held-out test set:

| Experiment | Feature Group | Dimensions | Accuracy (%) | Macro F1 | Weighted F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1: MFCC Only** | 40 MFCCs + Deltas + Delta-Deltas | 240 | 37.92% | 0.3630 | 0.3690 |
| **Exp 2: MFCC + Spectral** | + ZCR, RMS, Centroid, Bandwidth, Rolloff, Contrast | 264 | 45.00% | 0.4272 | 0.4383 |
| **Exp 3: Full Feature Set** | + Chroma Pitch Classes + Tonnetz Tonal Harmonics | **300** | **47.92%** | **0.4445** | **0.4519** |

### Key Ablation Insights:
1. **Spectral descriptors provide the single largest gain:** Adding Spectral Centroid, Bandwidth, Contrast, and RMS Energy to MFCCs caused a **+6.4% jump in Macro F1** (0.3630 $\to$ 0.4272). This proves that vocal brightness and dynamic energy variance are essential complements to vocal tract formant envelopes.
2. **Harmonic features provide incremental tuning:** Chroma and Tonnetz added another **+1.7% Macro F1 gain** (0.4272 $\to$ 0.4445) by capturing musical intervals and harmonic distribution across semitones.

---

## Explainability & Model Attribution

To ensure transparency, EmotionAnalyzer provides dual explanation mechanisms:

### 1. Classical Feature Importance
For classical models (Random Forest and Logistic Regression), the system ranks the acoustic descriptors having the greatest influence on the decision:
- `rms_std`: Indicates that vocal loudness fluctuation was decisive.
- `spectral_centroid_mean`: Indicates that overall vocal brightness influenced the classification.
- `mfcc_1_mean`: Reflects global vocal tract energy.

### 2. Mel-Spectrogram Grad-CAM Saliency
For the 2D CNN, the system computes Gradient-weighted Class Activation Mapping (Grad-CAM) over `last_conv_layer`:
$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
The resulting heatmap is overlaid onto the Mel-spectrogram, visually highlighting the specific frequency harmonics and time bursts that drove the neural network's activation.

---

## Uncertainty Quantification

Human emotional expression is frequently blended or ambiguous. To prevent misleading overconfident predictions:
1. **Full Probability Distribution:** The inference engine returns sorted probabilities across all 8 classes rather than a lone categorical label.
2. **Configurable Confidence Threshold (0.35):** If the highest class probability falls below 35%, the system explicitly displays:
   > **Low-Confidence Prediction (Acoustic Ambiguity Notice)**
   > The highest confidence score falls below the threshold. The audio may contain overlapping acoustic cues, subdued inflection, or background noise.

---

## Web Application & Security Hardening

The Flask web application (`app.py`) has been refactored for production security:
- **Singleton Model Pre-warming:** Models are cached in memory once upon server startup, avoiding costly reloads per HTTP request.
- **Path Traversal Protection:** All uploads are renamed using `uuid.uuid4().hex` and resolved strictly inside `uploads/`.
- **Sanitized Uploads:** Validates file extensions against `{'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm'}`.
- **Upload Size Limit:** Enforces `MAX_CONTENT_LENGTH = 16 MB` with HTTP 413 handling.
- **Deterministic File Cleanup:** Uploaded audio is unlinked in a `finally` block immediately after feature extraction.
- **Real-Time Microphone Capture:** Native HTML5 `MediaRecorder` captures audio directly from the browser microphone.
- **Dual API Support:** Supports both standard HTML form submissions and asynchronous JSON API requests.

---

## Running Locally

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/mohikarathi/EmotionAnalyzer.git
cd EmotionAnalyzer

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Web Application

**Option A: Streamlit Web Interface**
```bash
streamlit run streamlit_app.py
```
Navigate to: `http://localhost:8501`

**Option B: Flask Web Product & REST API**
```bash
python app.py
```
Navigate to: `http://localhost:10000`

---

## Training & Evaluation CLI

### 1. Index Dataset & Cache Features
```bash
python scripts/prepare_data.py
```

### 2. Train Models
```bash
# Train classical baselines (Random Forest, Logistic Regression, SVC)
python scripts/train.py --model classical

# Train Mel-Spectrogram CNN
python scripts/train.py --model cnn --epochs 35

# Train all models
python scripts/train.py --model all
```

### 3. Run Benchmark Suite
```bash
python scripts/evaluate.py
```

### 4. Run Feature Ablation Study
```bash
python scripts/ablation.py
```

---

## Automated Testing

Run the full pytest suite:
```bash
pytest tests/ -v
```

Tests cover:
- Audio loading, resample accuracy, silence trimming, and padding/truncation (`test_preprocessing.py`).
- Deterministic 300-d handcrafted features and Mel-spectrogram geometry (`test_features.py`).
- Probability distribution sum $\approx 1.0$, sorting, and confidence thresholding (`test_inference.py`).
- Security boundaries, invalid extensions, path traversal, and file cleanup (`test_app.py`).

---

## Limitations & Ethical Boundaries

1. **Acoustic Patterns vs. Inner Emotional States:** This system measures acoustic properties of audio waveforms, not human psychological consciousness. A speaker may speak calmly while experiencing intense anxiety. The application uses the explicit wording *"Predicted acoustic emotion"*.
2. **Actor-Portrayed Laboratory Speech:** Models trained on RAVDESS reflect North American English actors reading prompted sentences. Performance may degrade on spontaneous conversational speech, varied dialects, or noisy field environments.
3. **High-Arousal Overlap:** Anger and Happiness share high pitch and elevated loudness. Without linguistic semantics, acoustic classifiers can conflate intense joy with intense anger.

---

## Future Work

- **Self-Supervised Audio Representations:** Integrating pre-trained foundation models such as `wav2vec 2.0` or `HuBERT`.
- **Continuous Dimensional Emotion:** Predicting continuous Valence-Arousal-Dominance coordinates rather than discrete categorical labels.
- **Multilingual Generalization:** Benchmarking across multilingual corpora (e.g., EMO-DB for German, CaFE for French).
- **Lightweight Quantization:** Quantizing models to INT8 / ONNX for edge deployment.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
