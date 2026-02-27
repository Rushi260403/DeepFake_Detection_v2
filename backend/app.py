from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import gdown
from tensorflow.keras.models import load_model

# ==============================
# Flask App Config
# ==============================

app = Flask(
    __name__,
    template_folder="../ui/templates",
    static_folder="../ui/static"
)

# ==============================
# Model Download Section
# ==============================

MODEL_PATH = "model/face_classifier_balanced.h5"
MODEL_URL = "https://drive.google.com/uc?id=1Uk7jaDpOT6Wg3hdsB_yOO_ssnfJ3XZ7-"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    os.makedirs("model", exist_ok=True)
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ==============================
# Upload Folder
# ==============================

UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# Routes
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():

    try:
        file = request.files["video"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # -------- Frame Extraction --------
        cap = cv2.VideoCapture(filepath)

        fake_count = 0
        real_count = 0
        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % 15 == 0:

                # Resize frame for model
                resized = cv2.resize(frame, (224, 224))
                normalized = resized / 255.0
                input_frame = np.expand_dims(normalized, axis=0)

                # -------- Model Prediction --------
                prediction = model.predict(input_frame, verbose=0)[0][0]

                if prediction > 0.5:
                    fake_count += 1
                else:
                    real_count += 1

            frame_number += 1

        cap.release()

        total = fake_count + real_count

        if total == 0:
            return jsonify({"error": "No frames processed"}), 400

        fake_percent = round((fake_count / total) * 100, 2)
        real_percent = round((real_count / total) * 100, 2)

        final_result = "FAKE" if fake_percent > real_percent else "REAL"
        confidence = max(fake_percent, real_percent)

        return jsonify({
            "result": final_result,
            "confidence": confidence,
            "fake_percent": fake_percent,
            "real_percent": real_percent,
            "fake_count": fake_count,
            "real_count": real_count
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Video processing failed"}), 500


# ==============================
# Run Server
# ==============================

if __name__ == "__main__":
    app.run()