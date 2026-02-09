import os
import random
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE_PATH = os.path.join(PROJECT_ROOT, "dataset", "frames")
TARGET_PATH = os.path.join(PROJECT_ROOT, "dataset", "balanced_frames")

REAL_SRC = os.path.join(SOURCE_PATH, "real")
FAKE_SRC = os.path.join(SOURCE_PATH, "fake")

REAL_DST = os.path.join(TARGET_PATH, "real")
FAKE_DST = os.path.join(TARGET_PATH, "fake")

os.makedirs(REAL_DST, exist_ok=True)
os.makedirs(FAKE_DST, exist_ok=True)

real_images = os.listdir(REAL_SRC)
fake_images = os.listdir(FAKE_SRC)

real_count = len(real_images)
fake_sample = random.sample(fake_images, real_count)

print(f"Balancing dataset to {real_count} images per class")

# Copy REAL
for img in real_images:
    shutil.copy(
        os.path.join(REAL_SRC, img),
        os.path.join(REAL_DST, img)
    )

# Copy FAKE (downsampled)
for img in fake_sample:
    shutil.copy(
        os.path.join(FAKE_SRC, img),
        os.path.join(FAKE_DST, img)
    )

print("✅ Dataset balancing complete")
