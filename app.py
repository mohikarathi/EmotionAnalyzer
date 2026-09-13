"""EmotionAnalyzer Flask Web Application.

A secure, production-hardened web interface for Explainable Speech Emotion Recognition.
Provides audio upload and microphone recording with acoustic emotion inference,
uncertainty estimation, Mel-spectrogram rendering, and model attribution.
"""

import logging
import os
from pathlib import Path
import uuid
from typing import Optional

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename

from src.audio.preprocessing import preprocess_audio
from src.audio.visualization import (
    analyze_spectrogram_cues,
    compute_waveform_points,
    generate_melspectrogram_base64,
)
from src.config import (
    ALLOWED_EXTENSIONS,
    CONFIDENCE_THRESHOLD,
    FLASK_DEBUG,
    HOST,
    MAX_CONTENT_LENGTH,
    PORT,
    PROJECT_ROOT,
    UPLOADS_DIR,
)
from src.models.inference import get_inference_engine

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EmotionAnalyzer.app")

# Initialize Flask application
app = Flask(__name__, root_path=str(PROJECT_ROOT))
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "emotion-analyzer-secure-key-2026")
app.config["UPLOAD_FOLDER"] = str(UPLOADS_DIR)

# Ensure upload directory exists
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Pre-load inference engine at application startup
logger.info("Pre-warming inference engine...")
engine = get_inference_engine()
logger.info("Inference engine initialized successfully.")


def is_allowed_file(filename: str) -> bool:
    """Validate that the file extension is among permitted audio formats."""
    suffix = Path(filename).suffix.lower()
    return suffix in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def home():
    """Render the main application interface."""
    return render_template("index.html")


@app.route("/about", methods=["GET"])
def about():
    """Render the 'How It Works' technical explanation page."""
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Secure prediction endpoint supporting both form POST and AJAX requests."""
    # 1. Validate file presence in request
    if "audio" not in request.files:
        error_msg = "No audio file was included in the request."
        if request.headers.get("Accept") == "application/json" or request.is_json:
            return jsonify({"error": error_msg}), 400
        flash(error_msg, "error")
        return redirect(url_for("home"))

    file = request.files["audio"]

    # 2. Check for empty filename selection
    if not file or file.filename == "":
        error_msg = "Please select or record a valid audio file."
        if request.headers.get("Accept") == "application/json" or request.is_json:
            return jsonify({"error": error_msg}), 400
        flash(error_msg, "error")
        return redirect(url_for("home"))

    # 3. Check allowed extension
    original_filename = secure_filename(file.filename) or "recording.wav"
    if not is_allowed_file(original_filename):
        error_msg = f"Unsupported audio format. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        if request.headers.get("Accept") == "application/json" or request.is_json:
            return jsonify({"error": error_msg}), 400
        flash(error_msg, "error")
        return redirect(url_for("home"))

    # 4. Generate collision-proof unique temporary filename to prevent path traversal
    ext = Path(original_filename).suffix.lower()
    if not ext:
        ext = ".wav"
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    temp_path = UPLOADS_DIR / unique_filename

    # Ensure path stays strictly inside UPLOADS_DIR
    if not temp_path.resolve().parent == UPLOADS_DIR.resolve():
        return jsonify({"error": "Security violation: Invalid upload destination."}), 400

    try:
        # Save temporary file
        file.save(str(temp_path))

        # 5. Selected model (default to hybrid / best available)
        requested_model = request.form.get("model", "hybrid")

        # 6. Perform inference
        prediction_result = engine.predict(temp_path, model_name=requested_model)

        # 7. Generate visualization data & acoustic spectrogram cues
        y, sr = preprocess_audio(temp_path)
        waveform_points = compute_waveform_points(y, num_points=100)
        melspec_b64 = generate_melspectrogram_base64(y, sr=sr)
        spectrogram_cues = analyze_spectrogram_cues(y, sr=sr, predicted_emotion=prediction_result["predicted_emotion"])

        # 8. Check for optional model explanation
        explanation_data = None
        try:
            from src.models.explainability import explain_prediction
            explanation_data = explain_prediction(temp_path, model_name=requested_model)
        except Exception as expl_err:
            logger.debug(f"Explainability not available or skipped: {expl_err}")

        response_payload = {
            "success": True,
            "filename": original_filename,
            "prediction": prediction_result["predicted_emotion"],
            "confidence": prediction_result["confidence"],
            "is_low_confidence": prediction_result["is_low_confidence"],
            "confidence_threshold": prediction_result["confidence_threshold"],
            "probabilities": prediction_result["probabilities"],
            "model_used": prediction_result["model_used"],
            "waveform": waveform_points,
            "melspectrogram": melspec_b64,
            "spectrogram_cues": spectrogram_cues,
            "explanation": explanation_data
        }

        # Return JSON if requested by client (e.g. from AJAX/recorder)
        if request.headers.get("Accept") == "application/json" or request.is_json:
            return jsonify(response_payload)

        # Render HTML result template
        return render_template(
            "result.html",
            result=response_payload,
            filename=original_filename
        )

    except Exception as exc:
        logger.error(f"Inference error during processing of {original_filename}: {exc}", exc_info=True)
        user_error = f"Audio processing error: {str(exc)}"
        if request.headers.get("Accept") == "application/json" or request.is_json:
            return jsonify({"error": user_error}), 400
        flash(user_error, "error")
        return redirect(url_for("home"))

    finally:
        # 9. Deterministic cleanup: Ensure temporary upload is always deleted
        if temp_path.exists():
            try:
                temp_path.unlink()
                logger.debug(f"Cleaned up temporary upload: {temp_path.name}")
            except OSError as cleanup_err:
                logger.warning(f"Could not remove temp file {temp_path}: {cleanup_err}")


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle files exceeding MAX_CONTENT_LENGTH."""
    msg = f"Audio file is too large. Maximum allowed size is {MAX_CONTENT_LENGTH // (1024 * 1024)} MB."
    if request.headers.get("Accept") == "application/json" or request.is_json:
        return jsonify({"error": msg}), 413
    flash(msg, "error")
    return redirect(url_for("home")), 413


@app.errorhandler(404)
def not_found(error):
    """Handle missing pages gracefully."""
    return render_template("index.html"), 404


@app.errorhandler(500)
def server_error(error):
    """Handle unexpected server exceptions."""
    logger.error(f"Unhandled server error: {error}")
    msg = "An internal server error occurred while processing your request."
    if request.headers.get("Accept") == "application/json" or request.is_json:
        return jsonify({"error": msg}), 500
    flash(msg, "error")
    return redirect(url_for("home")), 500


if __name__ == "__main__":
    logger.info(f"Starting EmotionAnalyzer on {HOST}:{PORT} (debug={FLASK_DEBUG})")
    app.run(host=HOST, port=PORT, debug=FLASK_DEBUG)
