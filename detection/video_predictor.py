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
    PROJECT_ROOT, "model", "face_classifier_balanced.h5"
)

# ==================================================
# LOAD MODELS
# ==================================================
face_model = tf.keras.models.load_model(MODEL_PATH)
yolo = YOLO("yolov8n.pt")

IMG_SIZE = 224
FRAME_SKIP = 10

# Thresholds (from evaluation)
FAKE_THRESHOLD = 0.60
REAL_THRESHOLD = 0.40


# ==================================================
# FACE PREDICTION (CORRECT)
# ==================================================
def predict_face(face_img):

    face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_img = face_img.astype("float32") / 255.0
    face_img = np.expand_dims(face_img, axis=0)

    real_prob = face_model.predict(face_img, verbose=0)[0][0]

    # REAL = class 1
    # FAKE = class 0

    fake_prob = 1.0 - real_prob

    return fake_prob, real_prob


# ==================================================
# VIDEO PREDICTION
# ==================================================
def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)

    fake_probs = []
    real_probs = []

    frame_count = 0

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

                fake_prob, real_prob = predict_face(face)

                fake_probs.append(fake_prob)
                real_probs.append(real_prob)

    cap.release()

    # ==================================================
    # NO FACE FOUND
    # ==================================================

    if len(fake_probs) < 5:
        return "FACE NOT FOUND", 0.0

    fake_probs = np.array(fake_probs)
    real_probs = np.array(real_probs)

    mean_fake = fake_probs.mean()
    mean_real = real_probs.mean()

    strong_fake_ratio = np.mean(fake_probs >= FAKE_THRESHOLD)
    strong_real_ratio = np.mean(real_probs >= FAKE_THRESHOLD)

    # ==================================================
    # FINAL DECISION LOGIC (CORRECT & STABLE)
    # ==================================================

    # FAKE
    if mean_fake >= 0.55 or strong_fake_ratio >= 0.30:

        confidence = min(95.0, mean_fake * 100)

        return "FAKE", confidence

    # REAL
    if mean_real >= 0.55 or strong_real_ratio >= 0.30:

        confidence = min(95.0, mean_real * 100)

        return "REAL", confidence

    # UNCERTAIN
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
