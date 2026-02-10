import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# ==================================================
# PATHS
# ==================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "model", "face_classifier_finetuned.h5"
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT, "dataset", "balanced_frames"
)

REAL_PATH = os.path.join(DATASET_PATH, "real")
FAKE_PATH = os.path.join(DATASET_PATH, "fake")

IMG_SIZE = 224

print("📁 MODEL:", MODEL_PATH)
print("📁 DATASET:", DATASET_PATH)

# ==================================================
# LOAD MODEL
# ==================================================
model = tf.keras.models.load_model(MODEL_PATH)

# ==================================================
# THRESHOLDS
# ==================================================
FAKE_THRESHOLD = 0.7
REAL_THRESHOLD = 0.4

y_true, y_pred = [], []

# ==================================================
# EVALUATION FUNCTION
# ==================================================
def evaluate_folder(folder, true_label):
    total = 0
    used = 0

    for img in os.listdir(folder):
        if not img.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(folder, img)
        image = cv2.imread(img_path)
        if image is None:
            continue

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        prob = model.predict(image, verbose=0)[0][0]

        if prob >= FAKE_THRESHOLD:
            y_true.append(true_label)
            y_pred.append(1)
            used += 1
        elif prob <= REAL_THRESHOLD:
            y_true.append(true_label)
            y_pred.append(0)
            used += 1
        # else: UNCERTAIN → ignored

        total += 1
        if total % 2000 == 0:
            print(f"⏳ Processed {total} images...")

    print(f"✅ {os.path.basename(folder)} → used {used}/{total}")

# ==================================================
# RUN EVALUATION
# ==================================================
print("\n🔍 Evaluating REAL faces...")
evaluate_folder(REAL_PATH, 0)

print("\n🔍 Evaluating FAKE faces...")
evaluate_folder(FAKE_PATH, 1)

# ==================================================
# METRICS
# ==================================================
cm = confusion_matrix(y_true, y_pred)
report = classification_report(
    y_true, y_pred, target_names=["REAL", "FAKE"]
)

print("\n📊 CONFUSION MATRIX")
print(cm)

print("\n📈 CLASSIFICATION REPORT")
print(report)

tn, fp, fn, tp = cm.ravel()

print(f"\n⚠️ FALSE POSITIVE RATE (REAL → FAKE): {fp / (fp + tn):.3f}")
print(f"⚠️ FALSE NEGATIVE RATE (FAKE → REAL): {fn / (fn + tp):.3f}")

print("\n✅ STEP 4 — THRESHOLD EVALUATION COMPLETE")
