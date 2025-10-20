from flask import Flask, render_template, request, redirect, url_for
import os
from utils import predict_emotion

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["GET"])
def upload():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return redirect(url_for("upload_get"))
    file = request.files["audio"]
    if file.filename == "":
        return redirect(url_for("upload_get"))

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    # Use the real model prediction
    emotion = predict_emotion(save_path)

    return render_template("result.html", emotion=emotion, filename=file.filename)

if __name__ == "__main__":
    app.run(debug=True)
