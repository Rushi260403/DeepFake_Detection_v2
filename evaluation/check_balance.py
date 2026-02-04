import os

BASE_DIR = os.path.join("..", "dataset", "frames")

for cls in ["real", "fake"]:
    path = os.path.join(BASE_DIR, cls)

    if not os.path.exists(path):
        print(f"❌ Missing folder: {path}")
        continue

    count = len([
        f for f in os.listdir(path)
        if f.lower().endswith((".jpg", ".png"))
    ])

    print(f"✅ {cls.upper()} images: {count}")
