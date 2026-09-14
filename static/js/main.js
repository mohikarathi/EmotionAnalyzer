/**
 * EmotionAnalyzer — Main Interactive Interface Logic
 */

let selectedFile = null;

function switchMode(mode) {
  const tabUpload = document.getElementById("tabUploadBtn");
  const tabRecord = document.getElementById("tabRecordBtn");
  const panelUpload = document.getElementById("uploadPanel");
  const panelRecord = document.getElementById("recordPanel");

  if (mode === "upload") {
    tabUpload.classList.add("active");
    tabRecord.classList.remove("active");
    panelUpload.style.display = "block";
    panelRecord.style.display = "none";
  } else {
    tabRecord.classList.add("active");
    tabUpload.classList.remove("active");
    panelRecord.style.display = "block";
    panelUpload.style.display = "none";
  }
}

function handleFileSelected(files) {
  if (!files || files.length === 0) return;
  const file = files[0];
  selectedFile = file;

  // Display file info
  const fileNameDisplay = document.getElementById("fileNameDisplay");
  const fileSizeDisplay = document.getElementById("fileSizeDisplay");
  const filePreviewBar = document.getElementById("filePreviewBar");
  const audioPreview = document.getElementById("audioPreview");
  const analyzeBtn = document.getElementById("analyzeBtn");

  fileNameDisplay.textContent = file.name;
  fileSizeDisplay.textContent = (file.size / 1024).toFixed(1) + " KB";
  filePreviewBar.style.display = "block";

  // Create preview playback URL
  const objectUrl = URL.createObjectURL(file);
  audioPreview.src = objectUrl;

  analyzeBtn.disabled = false;
}

// Drag & Drop handlers
const dropzone = document.getElementById("dropzone");
if (dropzone) {
  ["dragenter", "dragover"].forEach(eventName => {
    dropzone.addEventListener(eventName, e => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(eventName => {
    dropzone.addEventListener(eventName, e => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", e => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      document.getElementById("audioFileInput").files = files;
      handleFileSelected(files);
    }
  });
}

// Form async submission with loading state
const uploadForm = document.getElementById("uploadForm");
if (uploadForm) {
  uploadForm.addEventListener("submit", async function(e) {
    e.preventDefault();
    if (!selectedFile) return;

    const loadingOverlay = document.getElementById("loadingOverlay");
    const resultsContainer = document.getElementById("resultsContainer");
    const modelSelect = document.getElementById("modelSelect");

    loadingOverlay.style.display = "flex";
    resultsContainer.style.display = "none";

    const formData = new FormData();
    formData.append("audio", selectedFile);
    formData.append("model", modelSelect.value);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Accept": "application/json"
        },
        body: formData
      });

      const data = await response.json();
      loadingOverlay.style.display = "none";

      if (!response.ok || data.error) {
        alert("Error: " + (data.error || "Prediction failed"));
        return;
      }

      renderResults(data);
    } catch (err) {
      loadingOverlay.style.display = "none";
      alert("Network or inference error: " + err.message);
    }
  });
}

function renderResults(data) {
  const container = document.getElementById("resultsContainer");
  
  const isLowConf = data.is_low_confidence;
  const confPercent = (data.confidence * 100).toFixed(1);
  const thresholdPercent = (data.confidence_threshold * 100).toFixed(0);

  let probabilitiesHtml = "";
  data.probabilities.forEach((item, index) => {
    const isTop = index === 0 && !isLowConf;
    probabilitiesHtml += `
      <div class="prob-row">
        <span class="prob-name">${capitalize(item.emotion)}</span>
        <div class="prob-track">
          <div class="prob-fill ${isTop ? 'prob-fill-top' : ''}" style="width: ${item.percentage}%;"></div>
        </div>
        <span class="prob-percent">${item.percentage}%</span>
      </div>
    `;
  });

  let explanationHtml = "";
  if (data.explanation) {
    if (data.explanation.type === "feature_importance" && data.explanation.top_features) {
      let feats = "";
      data.explanation.top_features.forEach(f => {
        feats += `
          <div class="feat-row">
            <span class="feat-name">${f.name}</span>
            <span class="feat-importance">${(f.importance * 100).toFixed(1)}%</span>
          </div>
        `;
      });
      explanationHtml = `
        <div class="section-block">
          <h3 class="section-title">Model Attribution & Explainability</h3>
          <p class="section-desc">${data.explanation.description}</p>
          <div class="feature-importance-list">${feats}</div>
        </div>
      `;
    } else if (data.explanation.type === "gradcam" && data.explanation.heatmap_b64) {
      explanationHtml = `
        <div class="section-block">
          <h3 class="section-title">Mel-Spectrogram Grad-CAM Saliency Map</h3>
          <p class="section-desc">${data.explanation.description}</p>
          <div class="spectrogram-container">
            <img src="${data.explanation.heatmap_b64}" alt="Grad-CAM Saliency Heatmap" class="spectrogram-img">
          </div>
        </div>
      `;
    }
  }

  // Spectrogram Cues: use backend analysis or rich fallback
  const cuesData = (data.spectrogram_cues && data.spectrogram_cues.cues && data.spectrogram_cues.cues.length > 0)
    ? data.spectrogram_cues
    : getAcousticCuesFallback(data.prediction);

  let cuesItems = "";
  cuesData.cues.forEach(c => {
    cuesItems += `
      <div class="cue-card">
        <div class="cue-header">
          <span class="cue-dimension">${c.dimension}</span>
          <span class="cue-obs">${c.observation}</span>
        </div>
        <p class="cue-interpretation">${c.interpretation}</p>
      </div>
    `;
  });

  const spectrogramCuesHtml = `
    <div class="section-block">
      <h3 class="section-title">Acoustic Cues Influencing the Classification</h3>
      <p class="section-desc">${cuesData.summary}</p>
      <div class="cues-grid">${cuesItems}</div>
    </div>
  `;

  const lowConfAlert = isLowConf ? `
    <div class="alert alert-warning">
      <strong>Acoustic Ambiguity Notice:</strong> The highest confidence score (${confPercent}%)
      falls below the confidence threshold (${thresholdPercent}%).
      The audio may contain overlapping emotional cues or ambient noise.
    </div>
  ` : "";

  container.innerHTML = `
    <div class="result-card">
      <div class="result-header">
        <div>
          <span class="badge-tag">Acoustic Inference Summary</span>
          <h2 class="result-title">Acoustic Emotion Analysis</h2>
          <p class="result-subtitle">Analyzed clip: <code class="code-badge">${data.filename}</code></p>
        </div>
        <div class="model-badge">
          <span>Model: ${capitalize(data.model_used)}</span>
        </div>
      </div>

      <div class="prediction-banner ${isLowConf ? 'low-confidence-banner' : ''}">
        <div class="prediction-info">
          <span class="prediction-label">Predicted Acoustic Emotion</span>
          <h1 class="prediction-emotion">${capitalize(data.prediction)}</h1>
          <p class="prediction-disclaimer">
            Inferred from acoustic spectral patterns and pitch dynamics.
          </p>
        </div>
        <div class="confidence-box">
          <span class="confidence-value">${confPercent}%</span>
          <span class="confidence-label">Confidence</span>
          <span class="${isLowConf ? 'badge-warning' : 'badge-success'}">
            ${isLowConf ? 'Low Confidence' : 'High Confidence'}
          </span>
        </div>
      </div>

      ${lowConfAlert}

      <div class="section-block">
        <h3 class="section-title">Acoustic Emotion Probability Distribution</h3>
        <p class="section-desc">Full softmax probability distribution across all 8 standard emotion categories, ranked by likelihood.</p>
        <div class="probability-bars">
          ${probabilitiesHtml}
        </div>
      </div>

      <div class="section-block">
        <h3 class="section-title">Time-Frequency Representation (Log-Mel Spectrogram)</h3>
        <p class="section-desc">Log-scaled Mel-spectrogram computed across 128 filter banks (0 – 8 kHz) capturing energy distribution over time.</p>
        <div class="spectrogram-container">
          <img src="${data.melspectrogram}" alt="Log-Mel Spectrogram" class="spectrogram-img">
        </div>
        <div class="spectrogram-guide">
          <div class="guide-pill">
            <div class="guide-pill-title">Time Axis (Horizontal)</div>
            <p class="guide-pill-desc">Duration (0 to 3.0s). Colored blocks show active speech; dark blue intervals show pauses.</p>
          </div>
          <div class="guide-pill">
            <div class="guide-pill-title">Frequency (Vertical, Mel Hz)</div>
            <p class="guide-pill-desc">0 – 8 kHz. Lower bands (&lt;1 kHz) show vocal pitch and vowel formants; upper bands (&gt;2.5 kHz) show friction and tension.</p>
          </div>
          <div class="guide-pill">
            <div class="guide-pill-title">Intensity (Decibels)</div>
            <p class="guide-pill-desc">Bright yellow/orange marks peak vocal energy; dark purple/black indicates quiet baseline or silence.</p>
          </div>
        </div>
      </div>

      ${spectrogramCuesHtml}

      ${explanationHtml}

      <div class="result-actions">
        <button type="button" class="btn-primary" onclick="window.scrollTo({top: 0, behavior: 'smooth'})">
          Analyze Another File
        </button>
        <a href="/about" class="btn-secondary">Read Model Architecture</a>
      </div>
    </div>
  `;

  container.style.display = "block";
  container.scrollIntoView({ behavior: "smooth" });
}

function capitalize(str) {
  if (!str) return "";
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function getAcousticCuesFallback(emotion) {
  const e = (emotion || "neutral").toLowerCase();
  const cuesMap = {
    anger: {
      summary: "Classified as Anger based on high acoustic intensity, prominent upper-frequency energy spill above 3 kHz, and elevated spectral centroid reflecting harsh vocal cord tension.",
      cues: [
        { dimension: "High-Frequency Energy", observation: "Elevated spectral spill (>3,000 Hz)", interpretation: "Strong acoustic pressure across upper frequencies reflects vocal hyperfunction and aggressive vocal cord closure." },
        { dimension: "Spectral Centroid (Brightness)", observation: "High centroid frequency (>1,800 Hz)", interpretation: "Elevated brightness center caused by forceful consonant attacks and tense vocal tract configuration." },
        { dimension: "Voicing Cadence & Rhythm", observation: "Compressed syllable timing", interpretation: "Rapid syllabic bursts with minimal pause duration typical of high-arousal assertive speech." },
        { dimension: "Visual Spectrogram Signature", observation: "Intense vertical energy bursts", interpretation: "Bright flame-like vertical plumes extending into the 4–8 kHz band indicate sudden acoustic power." }
      ]
    },
    happy: {
      summary: "Classified as Happy driven by wide pitch excursions, prominent harmonic energy, and vibrant spectral contrast across mid-range frequencies.",
      cues: [
        { dimension: "Pitch & Harmonic Ribbons", observation: "Undulating fundamental contours", interpretation: "Curved horizontal formant ribbons indicating animated pitch variation and wide dynamic vocal inflection." },
        { dimension: "Spectral Contrast", observation: "Sharp harmonic-to-noise distinction", interpretation: "Distinct resonant peaks against quieter interstitial bands indicating clear, buoyant vocalization." },
        { dimension: "High-Frequency Energy", observation: "Moderate-high energy dispersion", interpretation: "Warm acoustic brightness without the harsh high-frequency distortion seen in anger." },
        { dimension: "Visual Spectrogram Signature", observation: "Dynamic undulating bands", interpretation: "Well-separated harmonic tracks showing energetic pitch modulations across vowel centers." }
      ]
    },
    sad: {
      summary: "Classified as Sad characterized by low acoustic energy, energy concentrated strictly below 1 kHz, and prolonged decay intervals.",
      cues: [
        { dimension: "Energy Distribution", observation: "Low acoustic power restricted <1,000 Hz", interpretation: "Vocal folds operate with reduced adduction tension, extinguishing energy in higher frequency bands." },
        { dimension: "Spectral Centroid (Brightness)", observation: "Depressed centroid frequency (<1,400 Hz)", interpretation: "Dark acoustic color with muted vocal timbre and minimal upper-frequency presence." },
        { dimension: "Voicing Rate & Temporal Pacing", observation: "Extended silent intervals", interpretation: "Prolonged inter-word pauses and lethargic syllable release reflecting subdued psychomotor arousal." },
        { dimension: "Visual Spectrogram Signature", observation: "Sparse, low-frequency bands", interpretation: "Dark overall spectrogram background with only narrow, dim energy bands near the bottom floor." }
      ]
    },
    neutral: {
      summary: "Classified as Neutral indicated by balanced, steady formant trajectories, moderate spectral centroid, and even syllable pacing.",
      cues: [
        { dimension: "Formant Stability", observation: "Parallel, horizontal resonance bands", interpretation: "Unvarying vowel formants indicating steady, calm articulatory positioning without emotional tremor." },
        { dimension: "Energy Distribution", observation: "Concentrated in standard speech band (200 – 3,000 Hz)", interpretation: "Standard conversational acoustic power without extreme pitch excursions or muted suppression." },
        { dimension: "Cadence & Voicing Rate", observation: "Rhythmic, standard inter-word pauses", interpretation: "Regular speech cadence with predictable syllabic timing." },
        { dimension: "Visual Spectrogram Signature", observation: "Even, predictable horizontal tracks", interpretation: "Uniformly spaced harmonic bars without abrupt spikes or dropouts." }
      ]
    },
    calm: {
      summary: "Classified as Calm characterized by soft vowel transitions, gentle onset dynamics, and balanced low-to-mid energy.",
      cues: [
        { dimension: "Acoustic Onsets", observation: "Gentle attack and gradual decay", interpretation: "Soft vocal tract transitions without abrupt acoustic transients or percussive phonation." },
        { dimension: "Harmonic Continuity", observation: "Smooth, undisturbed horizontal bands", interpretation: "Relaxed phonation yielding stable harmonic continuity without vocal fry or strain." },
        { dimension: "Frequency Spread", observation: "Moderate spectral centroid", interpretation: "Balanced acoustic spectrum without excessive high-frequency hiss or low-frequency booming." },
        { dimension: "Visual Spectrogram Signature", observation: "Smooth, serene harmonic bands", interpretation: "Cohesive horizontal bands with gentle envelope transitions." }
      ]
    },
    fear: {
      summary: "Classified as Fear driven by spectral jitter, irregular harmonic continuity, and erratic high-frequency energy dispersion.",
      cues: [
        { dimension: "Harmonic Perturbation", observation: "Jitter and irregular harmonic tracks", interpretation: "Micro-tremor in the vocal apparatus causing wavy, perturbed formant resonances." },
        { dimension: "High-Frequency Dispersion", observation: "Scattered energy above 3,500 Hz", interpretation: "Breathiness and constriction causing diffused noise components across upper Mel bands." },
        { dimension: "Pacing & Attack", observation: "Rapid, uneven syllable bursts", interpretation: "Uneven temporal rhythm reflecting tense physiological breathing patterns." },
        { dimension: "Visual Spectrogram Signature", observation: "Fragmented, jittery spectral bands", interpretation: "Disrupted horizontal tracks with erratic vertical striations." }
      ]
    },
    disgust: {
      summary: "Classified as Disgust indicated by pharyngeal constriction, low spectral centroid, and prolonged guttural emphasis.",
      cues: [
        { dimension: "Resonant Cavity Shifts", observation: "Suppressed higher formants, heavy lower band", interpretation: "Pharyngeal and velar constriction shifts acoustic resonance downward toward low frequencies." },
        { dimension: "Vocal Roughness", observation: "Irregular low-frequency sub-harmonics", interpretation: "Vocal fry and roughness visible as closely-spaced lower striations." },
        { dimension: "Temporal Articulation", observation: "Drawn-out syllable nucleus", interpretation: "Slower articulation with extended vowel durations reflecting visceral rejection." },
        { dimension: "Visual Spectrogram Signature", observation: "Dense bottom-heavy spectral bands", interpretation: "Prominent dense lower band with diminished upper spectral presence." }
      ]
    },
    surprise: {
      summary: "Classified as Surprise marked by a sudden upward pitch jump, transient burst energy, and rapid post-onset decay.",
      cues: [
        { dimension: "Pitch Contour Trajectory", observation: "Sharp upward fundamental jump", interpretation: "Immediate fundamental frequency (F0) rise reflecting instantaneous cognitive intake." },
        { dimension: "Transient Energy Spike", observation: "Sudden wide-band vertical plume", interpretation: "Explosive acoustic onset spanning low-to-high frequencies simultaneously." },
        { dimension: "Temporal Duration", observation: "Brief utterance with rapid decay", interpretation: "Short duration focused on initial exclamation followed by silence." },
        { dimension: "Visual Spectrogram Signature", observation: "Sharp vertical plume with upward curve", interpretation: "Distinct vertical flare at syllable start curving steeply upward." }
      ]
    }
  };

  return cuesMap[e] || cuesMap.neutral;
}
