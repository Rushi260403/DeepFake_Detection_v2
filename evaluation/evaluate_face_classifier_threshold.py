import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "balanced_frames")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

# ==================================================
# DATA GENERATORS
# ==================================================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# ==================================================
# MODEL (STEP 2 — BALANCED TRAINING)
# ==================================================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Class weights (REAL boosted)
class_weight = {
    0: 5.7,  # REAL
    1: 1.0   # FAKE
}

print("\n🚀 STEP 2: Training with class weights...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weight
)

# ==================================================
# STEP 3 — FINE-TUNING BACKBONE
# ==================================================
print("\n🔧 STEP 3: Fine-tuning last 30 layers...")

for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=2,
    class_weight=class_weight
)

# ==================================================
# SAVE FINAL MODEL
# ==================================================
FINAL_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "model", "face_classifier_finetuned.h5"
)
model.save(FINAL_MODEL_PATH)

print("✅ Fine-tuned balanced model saved at:")
print(FINAL_MODEL_PATH)


