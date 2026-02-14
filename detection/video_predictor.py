import cv2
import os
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# ==================================================
# PATHS
# ==================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "model", "face_classifier_finetuned.h5"
)

# ==================================================
# LOAD MODELS
# ==================================================
face_model = tf.keras.models.load_model(MODEL_PATH)
yolo = YOLO("yolov8n.pt")

IMG_SIZE = 224
FRAME_SKIP = 10   # CPU friendly

# 🔥 OPTIMIZED THRESHOLDS (from Phase 3.3)
FAKE_THRESHOLD = 0.60
REAL_THRESHOLD = 0.35

# ==================================================
# FACE PREDICTION
# ==================================================
def predict_face(face_img):
    face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)

    real_prob = face_model.predict(face_img, verbose=0)[0][0]
    fake_prob = 1.0 - real_prob   # 🔥 CRITICAL FIX

    return fake_prob

# ==================================================
# VIDEO PREDICTION
# ==================================================
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    fake_probs = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        results = yolo(frame, conf=0.4)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                prob = predict_face(face)
                fake_probs.append(prob)

    cap.release()

    # 🚨 NO FACE FOUND
    if len(fake_probs) == 0:
        return "FACE NOT FOUND", 0.0

    # ==================================================
    # 🔥 VIDEO-LEVEL VOTING (CRITICAL FIX)
    # ==================================================
    fake_count = sum(p >= FAKE_THRESHOLD for p in fake_probs)
    total = len(fake_probs)
    fake_ratio = fake_count / total

    if fake_ratio >= 0.30:
        return "FAKE", fake_ratio * 100

    elif fake_ratio <= 0.10:
        return "REAL", (1 - fake_ratio) * 100

    else:
        return "UNCERTAIN", 50.0

# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":
    video_path = input("🎥 Enter video path: ").strip()

    if not os.path.exists(video_path):
        print("❌ Video path not found")
        exit()

    label, confidence = predict_video(video_path)

    print("\n🎯 FINAL VIDEO PREDICTION")
    print(f"Result     : {label}")
    print(f"Confidence : {confidence:.2f}%")