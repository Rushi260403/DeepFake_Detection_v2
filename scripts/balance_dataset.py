import os
import random
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SRC_REAL = os.path.join(PROJECT_ROOT, "dataset", "frames", "real")
SRC_FAKE = os.path.join(PROJECT_ROOT, "dataset", "frames", "fake")

DST_ROOT = os.path.join(PROJECT_ROOT, "dataset", "balanced_frames")
DST_REAL = os.path.join(DST_ROOT, "real")
DST_FAKE = os.path.join(DST_ROOT, "fake")

os.makedirs(DST_REAL, exist_ok=True)
os.makedirs(DST_FAKE, exist_ok=True)

real_imgs = os.listdir(SRC_REAL)
fake_imgs = os.listdir(SRC_FAKE)

real_count = len(real_imgs)

# 🔥 CRITICAL: limit fake frames to real count
fake_imgs = random.sample(fake_imgs, real_count)

print(f"REAL frames used : {real_count}")
print(f"FAKE frames used : {len(fake_imgs)}")

for img in real_imgs:
    shutil.copy(os.path.join(SRC_REAL, img), DST_REAL)

for img in fake_imgs:
    shutil.copy(os.path.join(SRC_FAKE, img), DST_FAKE)

print("✅ Balanced frame dataset created")
