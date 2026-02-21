import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from backend.video_service import analyze_video

# ==================================================
# PROJECT PATHS
# ==================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
FRAME_RESULT = os.path.join(PROJECT_ROOT, "backend", "static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_RESULT, exist_ok=True)

# ==================================================
# CREATE FLASK APP (MUST COME FIRST)
# ==================================================
app = Flask(
    __name__,
    template_folder="../ui/templates",
    static_folder="../ui/static"
)

# ==================================================
# ROUTES
# ==================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"})

    video = request.files["video"]

    video_path = os.path.join(UPLOAD_FOLDER, video.filename)

    video.save(video_path)

    prediction, confidence, frames = analyze_video(video_path)

    return jsonify({
        "result": prediction,
        "confidence": f"{confidence:.2f}%",
        "frames": frames
    })


from flask import send_from_directory

@app.route("/frame_results/<filename>")
def get_frame(filename):
    return send_from_directory(FRAME_RESULT, filename)


# ==================================================
# RUN SERVER
# ==================================================
if __name__ == "__main__":
    app.run(debug=True)