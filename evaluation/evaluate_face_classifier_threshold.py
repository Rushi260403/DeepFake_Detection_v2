import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# =================================================
# PATHS
# =================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "face_classifier_balanced.h5")
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "balanced_frames")

REAL_PATH = os.path.join(DATASET_PATH, "real")
FAKE_PATH = os.path.join(DATASET_PATH, "fake")

print("📁 MODEL PATH :", MODEL_PATH)
print("📁 DATASET    :", DATASET_PATH)

# =================================================
# LOAD MODEL
# =================================================
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224

# =================================================
# THRESHOLDS
# =================================================
FAKE_THRESHOLD = 0.60
REAL_THRESHOLD = 0.35

y_true = []
y_pred = []

# =================================================
# PREDICTION FUNCTION
# =================================================
def predict_folder(folder, true_label):

    total = 0

    for img_name in os.listdir(folder):

        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(folder, img_name)

        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        prob = model.predict(image, verbose=0)[0][0]

        # 🔥 ALWAYS assign prediction
        pred_label = 1 if prob >= 0.5 else 0

        y_true.append(true_label)
        y_pred.append(pred_label)

        total += 1

        if total % 2000 == 0:
            print(f"⏳ Processed {total} images...")

    print(f"✅ {os.path.basename(folder)} → used {total} images")


# =================================================
# RUN
# =================================================
print("\n🔍 Evaluating REAL faces...")
predict_folder(REAL_PATH, 0)

print("\n🔍 Evaluating FAKE faces...")
predict_folder(FAKE_PATH, 1)


# =================================================
# METRICS
# =================================================
cm = confusion_matrix(y_true, y_pred)

print("\n📊 CONFUSION MATRIX")
print(cm)

print("\n📈 CLASSIFICATION REPORT")
print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"]))


tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn)
fnr = fn / (fn + tp)

print(f"\n⚠️ FALSE POSITIVE RATE (REAL → FAKE): {fpr:.4f}")
print(f"⚠️ FALSE NEGATIVE RATE (FAKE → REAL): {fnr:.4f}")

print("\n✅ Evaluation Complete")
