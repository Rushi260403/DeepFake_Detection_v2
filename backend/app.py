import os
from flask import Flask, render_template, request, jsonify

from backend.video_service import analyze_video


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


app = Flask(
    __name__,
    template_folder="../ui/templates",
    static_folder="../ui/static"
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"})

    video = request.files["video"]

    video_path = os.path.join(UPLOAD_FOLDER, video.filename)

    video.save(video_path)

    prediction, confidence, frames = analyze_video(video_path)

    return jsonify({
        "result": prediction,
        "confidence": round(confidence, 2),
        "frames": frames
    })


if __name__ == "__main__":
    app.run(debug=True)