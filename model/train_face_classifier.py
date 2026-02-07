import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =================================================
# 🔧 PROJECT ROOT (ABSOLUTE & SAFE)
# =================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "frames")
REAL_PATH = os.path.join(DATASET_PATH, "real")
FAKE_PATH = os.path.join(DATASET_PATH, "fake")

print("📁 PROJECT ROOT :", PROJECT_ROOT)
print("📁 DATASET PATH :", DATASET_PATH)
print("📁 REAL PATH    :", REAL_PATH)
print("📁 FAKE PATH    :", FAKE_PATH)

# =================================================
# ❌ HARD VALIDATION (NO SILENT FAILURES)
# =================================================
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"❌ Missing folder: {DATASET_PATH}")

if not os.path.exists(REAL_PATH):
    raise FileNotFoundError(f"❌ Missing folder: {REAL_PATH}")

if not os.path.exists(FAKE_PATH):
    raise FileNotFoundError(f"❌ Missing folder: {FAKE_PATH}")

if len(os.listdir(REAL_PATH)) == 0 or len(os.listdir(FAKE_PATH)) == 0:
    raise RuntimeError("❌ REAL or FAKE folder is empty. Run face extraction first.")

# =================================================
# ⚙️ TRAINING CONFIG
# =================================================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# =================================================
# 🧠 MODEL (CPU FRIENDLY)
# =================================================
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False
for layer in base_model.layers[-15:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =================================================
# 🚀 TRAIN
# =================================================
class_weight = {
    0: 1.0,   # REAL
    1: 6.0    # FAKE
}

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weight
)

# =================================================
# 💾 SAVE MODEL
# =================================================
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "face_classifier.h5")
model.save(MODEL_PATH)

print(f"✅ TRAINING COMPLETE")
print(f"✅ MODEL SAVED AT: {MODEL_PATH}")
