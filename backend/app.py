from flask import Flask, render_template, request, jsonify
import os
import cv2
import random

app = Flask(__name__,
            template_folder="../ui/templates",
            static_folder="../ui/static")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():

    file = request.files["video"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # -------- Frame Extraction --------
    cap = cv2.VideoCapture(filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fake_count = 0
    real_count = 0

    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % 15 == 0:  # sample frames
            # -------- Dummy Prediction --------
            prediction = random.choice(["FAKE", "REAL"])

            if prediction == "FAKE":
                fake_count += 1
            else:
                real_count += 1

        frame_number += 1

    cap.release()

    total = fake_count + real_count

    fake_percent = round((fake_count / total) * 100, 2) if total != 0 else 0
    real_percent = round((real_count / total) * 100, 2) if total != 0 else 0

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

if __name__ == "__main__":
    app.run()