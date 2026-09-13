"""Unit and integration tests for Flask web application and security checks."""

import io
import pytest
import soundfile as sf
import numpy as np
from pathlib import Path

from app import app
from src.config import UPLOADS_DIR


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def valid_audio_stream():
    """Generate in-memory WAV audio BytesIO stream."""
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV")
    buf.seek(0)
    return buf


def test_home_route(client):
    """Verify home route returns 200 OK."""
    response = client.get("/")
    assert response.status_code == 200


def test_about_route(client):
    """Verify about route returns 200 OK."""
    response = client.get("/about")
    assert response.status_code == 200


def test_predict_no_file(client):
    """Verify missing audio file returns 400 when requesting JSON."""
    response = client.post("/predict", data={}, headers={"Accept": "application/json"})
    assert response.status_code == 400
    assert "No audio file" in response.get_json()["error"]


def test_predict_invalid_extension(client):
    """Verify unallowed file extensions (.txt, .exe) are blocked."""
    data = {"audio": (io.BytesIO(b"malicious script content"), "exploit.sh")}
    response = client.post("/predict", data=data, headers={"Accept": "application/json"})
    assert response.status_code == 400
    assert "Unsupported audio format" in response.get_json()["error"]


def test_predict_corrupt_audio(client):
    """Verify corrupt audio is handled gracefully with an error response."""
    data = {"audio": (io.BytesIO(b"corrupt audio header garbage"), "corrupt.wav")}
    response = client.post("/predict", data=data, headers={"Accept": "application/json"})
    assert response.status_code == 400
    assert "Audio processing error" in response.get_json()["error"]


def test_predict_valid_audio_json(client, valid_audio_stream):
    """Verify valid audio returns full JSON response with probabilities, waveform, and cleanup."""
    # Count files in uploads before
    files_before = list(UPLOADS_DIR.glob("*"))

    data = {"audio": (valid_audio_stream, "test_sample.wav")}
    response = client.post("/predict", data=data, headers={"Accept": "application/json"})

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert "prediction" in json_data
    assert "confidence" in json_data
    assert "probabilities" in json_data
    assert len(json_data["probabilities"]) == 8

    # Verify probability distribution sums close to 1.0
    total_prob = sum(item["score"] for item in json_data["probabilities"])
    assert np.isclose(total_prob, 1.0, atol=0.05)

    # Verify waveform points exist
    assert len(json_data["waveform"]) == 100

    # Verify Mel spectrogram base64 exists
    assert json_data["melspectrogram"].startswith("data:image/png;base64,")

    # Verify temporary file cleanup: no new files left behind in uploads
    files_after = list(UPLOADS_DIR.glob("*"))
    assert len(files_after) == len(files_before)
