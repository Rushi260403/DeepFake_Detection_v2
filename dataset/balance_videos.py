import os
import shutil
import random

# ==================================================
# PROJECT ROOT
# ==================================================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# ==================================================
# INPUT RAW VIDEOS  ✅ (Possibility B FIX)
# ==================================================
RAW_DATASET = os.path.join(PROJECT_ROOT, "dataset", "videos")
REAL_VIDEOS = os.path.join(RAW_DATASET, "real")
FAKE_VIDEOS = os.path.join(RAW_DATASET, "fake")

# ==================================================
# OUTPUT BALANCED VIDEOS
# ==================================================
BALANCED_DATASET = os.path.join(PROJECT_ROOT, "dataset", "balanced_videos")
BAL_REAL = os.path.join(BALANCED_DATASET, "real")
BAL_FAKE = os.path.join(BALANCED_DATASET, "fake")

os.makedirs(BAL_REAL, exist_ok=True)
os.makedirs(BAL_FAKE, exist_ok=True)

# ==================================================
# LIST VIDEO FILES
# ==================================================
real_files = [
    f for f in os.listdir(REAL_VIDEOS)
    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
]

fake_files = [
    f for f in os.listdir(FAKE_VIDEOS)
    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
]

# ==================================================
# BALANCE COUNT
# ==================================================
N = min(len(real_files), len(fake_files))
print(f"🔢 Balancing {N} REAL and {N} FAKE videos")

random.shuffle(real_files)
random.shuffle(fake_files)

real_selected = real_files[:N]
fake_selected = fake_files[:N]

# ==================================================
# COPY FILES
# ==================================================
for f in real_selected:
    shutil.copy(
        os.path.join(REAL_VIDEOS, f),
        os.path.join(BAL_REAL, f)
    )

for f in fake_selected:
    shutil.copy(
        os.path.join(FAKE_VIDEOS, f),
        os.path.join(BAL_FAKE, f)
    )

print("✅ STEP 1 COMPLETE — Balanced videos created")
