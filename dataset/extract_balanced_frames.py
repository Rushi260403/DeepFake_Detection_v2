import cv2
import os
import numpy as np
from ultralytics import YOLO

# ==================================================
# PATHS
# ==================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

VIDEO_ROOT = os.path.join(PROJECT_ROOT, "dataset", "balanced_videos")
OUT_ROOT = os.path.join(PROJECT_ROOT, "dataset", "balanced_frames")

REAL_VIDEOS = os.path.join(VIDEO_ROOT, "real")
FAKE_VIDEOS = os.path.join(VIDEO_ROOT, "fake")

OUT_REAL = os.path.join(OUT_ROOT, "real")
OUT_FAKE = os.path.join(OUT_ROOT, "fake")

os.makedirs(OUT_REAL, exist_ok=True)
os.makedirs(OUT_FAKE, exist_ok=True)

# ==================================================
# SETTINGS
# ==================================================
MAX_FRAMES_PER_VIDEO = 30   # 🔥 critical (controls balance)
IMG_SIZE = 224
FRAME_SKIP = 5

yolo = YOLO("yolov8n.pt")

# ==================================================
# FRAME EXTRACTION
# ==================================================
def extract(video_path, out_dir):
    cap = cv2.VideoCapture(video_path)
    saved = 0
    frame_id = 0

    while cap.isOpened() and saved < MAX_FRAMES_PER_VIDEO:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue

        results = yolo(frame, conf=0.4)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                fname = f"{os.path.basename(video_path)}_{saved}.jpg"
                cv2.imwrite(os.path.join(out_dir, fname), face)
                saved += 1

                if saved >= MAX_FRAMES_PER_VIDEO:
                    break

    cap.release()

# ==================================================
# RUN
# ==================================================
for v in os.listdir(REAL_VIDEOS):
    extract(os.path.join(REAL_VIDEOS, v), OUT_REAL)

for v in os.listdir(FAKE_VIDEOS):
    extract(os.path.join(FAKE_VIDEOS, v), OUT_FAKE)

print("✅ STEP 2 COMPLETE — Balanced frames extracted")

