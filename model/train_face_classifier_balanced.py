import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# ==================================================
# PATHS
# ==================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "balanced_frames")
MODEL_OUT = os.path.join(PROJECT_ROOT, "model", "face_classifier_balanced.h5")

# ==================================================
# SETTINGS
# ==================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_STAGE_1 = 6   # feature learning
EPOCHS_STAGE_2 = 4   # fine-tuning

# ==================================================
# DATA GENERATORS
# ==================================================
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# ==================================================
# MODEL — STAGE 1 (FROZEN BACKBONE)
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
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\n🚀 STAGE 1 — Training classifier head")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE_1
)

# ==================================================
# MODEL — STAGE 2 (FINE-TUNING)
# ==================================================
print("\n🔧 STAGE 2 — Fine-tuning last 30 layers")

for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # VERY LOW LR
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE_2
)

# ==================================================
# SAVE MODEL
# ==================================================
model.save(MODEL_OUT)
print(f"\n✅ STEP 3 COMPLETE — Model saved at:\n{MODEL_OUT}")
