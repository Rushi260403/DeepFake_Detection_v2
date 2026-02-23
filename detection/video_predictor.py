import cv2
import os
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import uuid

# ==================================================
# PATHS
# ==================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "face_classifier_balanced.h5")

# Folder to save detected frames for web display
OUTPUT_FRAMES_DIR = os.path.join(PROJECT_ROOT, "backend", "static", "results")

os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

# ==================================================
# LOAD MODELS
# ==================================================
print("Loading face classifier...")
face_model = tf.keras.models.load_model(MODEL_PATH)

print("Loading YOLO face detector...")
yolo = YOLO("yolov8n.pt")

IMG_SIZE = 224
FRAME_SKIP = 10

FAKE_THRESHOLD = 0.60
REAL_THRESHOLD = 0.40


# ==================================================
# FACE PREDICTION
# ==================================================
def predict_face(face_img):

    face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_img = face_img.astype("float32") / 255.0
    face_img = np.expand_dims(face_img, axis=0)

    real_prob = face_model.predict(face_img, verbose=0)[0][0]
    fake_prob = 1.0 - real_prob

    return fake_prob, real_prob


# ==================================================
# VIDEO PREDICTION (WEB VERSION)
# ==================================================
def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)

    fake_probs = []
    real_probs = []

    saved_frames = []

    frame_count = 0
    video_id = str(uuid.uuid4())[:8]

    print("Analyzing video...")

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

                # ==================================================
                # VISUAL FEATURES (Production UI Frame Styling)
                # ==================================================

                label = "fake" if fake_prob >= FAKE_THRESHOLD else "real"

                # Select border color
                if label == "fake":
                    color = (0, 0, 255)  # RED
                elif label == "real":
                    color = (0, 255, 0)  # GREEN
                else:
                    color = (0, 255, 255)  # YELLOW

                # Confidence %
                confidence_value = max(fake_prob, real_prob) * 100

                confidence_text = f"{label.upper()} {confidence_value:.1f}%"

                # Draw rectangle around detected face
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3
                )

                # Draw filled rectangle behind text
                (text_width, text_height), _ = cv2.getTextSize(
                    confidence_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )

                cv2.rectangle(
                    frame,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width, y1),
                    color,
                    -1
                )

                # Draw confidence text
                cv2.putText(
                    frame,
                    confidence_text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # Save styled frame
                frame_filename = f"{video_id}_{frame_count}_{label}.jpg"

                frame_path = os.path.join(
                    OUTPUT_FRAMES_DIR,
                    frame_filename
                )

                cv2.imwrite(frame_path, frame)

                # Save frame info for UI
                saved_frames.append({
                    "path": f"/static/results/{frame_filename}",
                    "fake_prob": float(fake_prob),
                    "real_prob": float(real_prob),
                    "label": label,
                    "confidence": float(confidence_value)
                })

    cap.release()

    # ==================================================
    # NO FACE FOUND
    # ==================================================
    if len(fake_probs) < 5:

        return {
            "prediction": "FACE NOT FOUND",
            "confidence": 0.0,
            "frames": []
        }

    fake_probs = np.array(fake_probs)
    real_probs = np.array(real_probs)

    mean_fake = fake_probs.mean()
    mean_real = real_probs.mean()

    strong_fake_ratio = np.mean(fake_probs >= FAKE_THRESHOLD)
    strong_real_ratio = np.mean(real_probs >= REAL_THRESHOLD)

    # ==================================================
    # FINAL DECISION
    # ==================================================
    if mean_fake >= 0.55 or strong_fake_ratio >= 0.30:

        return {
            "prediction": "FAKE",
            "confidence": float(mean_fake * 100),
            "frames": saved_frames
        }

    if mean_real >= 0.55 or strong_real_ratio >= 0.30:

        return {
            "prediction": "REAL",
            "confidence": float(mean_real * 100),
            "frames": saved_frames
        }

    return {
        "prediction": "UNCERTAIN",
        "confidence": 50.0,
        "frames": saved_frames
    }


# ==================================================
# TERMINAL TEST
# ==================================================
if __name__ == "__main__":

    video_path = input("Enter video path: ")

    result = predict_video(video_path)

    print("\nRESULT:", result["prediction"])
    print("CONFIDENCE:", result["confidence"])