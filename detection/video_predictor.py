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

    return face_model.predict(face_img, verbose=0)[0][0]

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

    if len(fake_probs) < 5:
        return "FACE NOT FOUND", 0.0

    return final_video_decision(fake_probs)

    # ==================================================
    # 🔥 VIDEO-LEVEL VOTING (CRITICAL FIX)
    # ==================================================
def final_video_decision(probs):
    probs = np.array(probs)

    mean = probs.mean()
    std = probs.std()

    real_frames = np.sum(probs >= 0.65)
    fake_frames = np.sum(probs <= 0.35)

    total = len(probs)

    real_ratio = real_frames / total
    fake_ratio = fake_frames / total

    # STRONG FAKE
    if fake_ratio >= 0.25 and mean <= 0.45:
        return "FAKE", min(95.0, (1 - mean) * 100)

    # STRONG REAL
    if real_ratio >= 0.60 and mean >= 0.55:
        return "REAL", min(95.0, mean * 100)

    # AI / AMBIGUOUS
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