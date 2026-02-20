import os
import numpy as np
import tensorflow as tf
import cv2
from sklearn.metrics import confusion_matrix, classification_report

# ==================================================
# PATH SETUP
# ==================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "face_classifier_balanced.h5"
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "balanced_frames"
)

REAL_PATH = os.path.join(DATASET_PATH, "real")
FAKE_PATH = os.path.join(DATASET_PATH, "fake")

IMG_SIZE = 224
BATCH_SIZE = 32

# ==================================================
# LOAD MODEL
# ==================================================
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# ==================================================
# GENERATOR FUNCTION (MEMORY SAFE)
# ==================================================
def image_generator(folder, label):

    files = os.listdir(folder)

    print(f"\nScanning {folder}")
    print(f"Found {len(files)} images")

    batch_images = []
    batch_labels = []

    for i, file in enumerate(files):

        path = os.path.join(folder, file)

        img = cv2.imread(path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0   # 🔥 FIX MEMORY
        batch_images.append(img)
        batch_labels.append(label)

        if len(batch_images) == BATCH_SIZE:

            yield np.array(batch_images), np.array(batch_labels)
            batch_images = []
            batch_labels = []

        if (i+1) % 2000 == 0:
            print(f"Processed {i+1} images...")

    if batch_images:
        yield np.array(batch_images), np.array(batch_labels)

# ==================================================
# PREDICT IN BATCHES
# ==================================================
y_true = []
y_pred = []

print("\nEvaluating REAL images...")

for X_batch, y_batch in image_generator(REAL_PATH, 0):

    probs = model.predict(X_batch, verbose=0)
    preds = (probs >= 0.5).astype(int).flatten()

    y_true.extend(y_batch)
    y_pred.extend(preds)

print("\nEvaluating FAKE images...")

for X_batch, y_batch in image_generator(FAKE_PATH, 1):

    probs = model.predict(X_batch, verbose=0)
    preds = (probs >= 0.5).astype(int).flatten()

    y_true.extend(y_batch)
    y_pred.extend(preds)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ==================================================
# RESULTS
# ==================================================
print("\nCONFUSION MATRIX")
cm = confusion_matrix(y_true, y_pred)
print(cm)

print("\nCLASSIFICATION REPORT")
print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"]))

tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn)
fnr = fn / (fn + tp)

print(f"\nFALSE POSITIVE RATE: {fpr:.3f}")
print(f"FALSE NEGATIVE RATE: {fnr:.3f}")

print("\n✅ Evaluation Complete (Memory Safe)")