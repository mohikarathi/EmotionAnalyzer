/**
 * EmotionAnalyzer — Browser Microphone Recording (MediaRecorder API)
 */

let mediaRecorder = null;
let audioChunks = [];
let recordTimerInterval = null;
let recordSeconds = 0;
let recordedAudioBlob = null;

async function startRecording() {
  audioChunks = [];
  recordSeconds = 0;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    
    // Choose supported MIME type
    const mimeTypes = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/mp4",
      ""
    ];
    let selectedMime = "";
    for (const mime of mimeTypes) {
      if (!mime || MediaRecorder.isTypeSupported(mime)) {
        selectedMime = mime;
        break;
      }
    }

    const options = selectedMime ? { mimeType: selectedMime } : {};
    mediaRecorder = new MediaRecorder(stream, options);

    mediaRecorder.ondataavailable = event => {
      if (event.data && event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = () => {
      stream.getTracks().forEach(track => track.stop());
      const blobType = selectedMime || "audio/webm";
      recordedAudioBlob = new Blob(audioChunks, { type: blobType });

      const recordedPreviewBar = document.getElementById("recordedPreviewBar");
      const recordedAudioPreview = document.getElementById("recordedAudioPreview");
      recordedAudioPreview.src = URL.createObjectURL(recordedAudioBlob);
      recordedPreviewBar.style.display = "block";
    };

    mediaRecorder.start(100); // 100ms slice chunks

    // UI state updates
    document.getElementById("startRecordBtn").style.display = "none";
    document.getElementById("stopRecordBtn").style.display = "inline-block";
    document.getElementById("recordPulse").classList.add("active");
    document.getElementById("recordedPreviewBar").style.display = "none";
    document.getElementById("recorderInstruction").textContent = "Recording active... Speak clearly.";

    // Start timer
    const timerDisplay = document.getElementById("recordTimer");
    timerDisplay.textContent = "00:00";
    recordTimerInterval = setInterval(() => {
      recordSeconds++;
      const mins = String(Math.floor(recordSeconds / 60)).padStart(2, "0");
      const secs = String(recordSeconds % 60).padStart(2, "0");
      timerDisplay.textContent = `${mins}:${secs}`;

      // Automatically cap recording at 10 seconds
      if (recordSeconds >= 10) {
        stopRecording();
      }
    }, 1000);

  } catch (err) {
    console.error("Microphone access error:", err);
    alert("Microphone access failed: " + err.message + "\nPlease grant microphone permissions in your browser.");
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }

  clearInterval(recordTimerInterval);
  document.getElementById("startRecordBtn").style.display = "inline-block";
  document.getElementById("stopRecordBtn").style.display = "none";
  document.getElementById("recordPulse").classList.remove("active");
  document.getElementById("recorderInstruction").textContent = "Recording complete. Review playback below or analyze.";
}

async function submitRecordedAudio() {
  if (!recordedAudioBlob) {
    alert("Please record audio before analyzing.");
    return;
  }

  const loadingOverlay = document.getElementById("loadingOverlay");
  const resultsContainer = document.getElementById("resultsContainer");
  const modelSelect = document.getElementById("modelSelect");

  loadingOverlay.style.display = "flex";
  resultsContainer.style.display = "none";

  const formData = new FormData();
  // Provide filename extension matching blob type
  const extension = recordedAudioBlob.type.includes("ogg") ? "ogg" : "webm";
  formData.append("audio", recordedAudioBlob, `mic_recording.${extension}`);
  formData.append("model", modelSelect ? modelSelect.value : "hybrid");

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
    alert("Inference failed: " + err.message);
  }
}
