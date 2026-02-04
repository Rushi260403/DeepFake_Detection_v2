import os
import cv2
from face_detector import YOLOFaceDetector

# =============================
# PATH SETUP (ABSOLUTE)
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
VIDEO_DIR = os.path.join(DATASET_DIR, "videos")
FRAME_DIR = os.path.join(DATASET_DIR, "frames")

FRAME_SKIP = 30  # 1 frame per second (CPU safe)
detector = YOLOFaceDetector()


def load_progress(progress_file):
    if not os.path.exists(progress_file):
        return set()
    with open(progress_file, "r") as f:
        return set(line.strip() for line in f.readlines())


def save_progress(progress_file, video_name):
    with open(progress_file, "a") as f:
        f.write(video_name + "\n")


def run_extraction(class_name):
    video_dir = os.path.join(VIDEO_DIR, class_name)
    output_dir = os.path.join(FRAME_DIR, class_name)
    progress_file = os.path.join(FRAME_DIR, f".progress_{class_name}.txt")

    os.makedirs(output_dir, exist_ok=True)

    processed_videos = load_progress(progress_file)

    print(f"\n📂 Class: {class_name}")
    print(f"📌 Already processed: {len(processed_videos)} videos")

    for video_file in os.listdir(video_dir):
        if not video_file.lower().endswith((".mp4", ".avi", ".mov")):
            continue

        if video_file in processed_videos:
            print(f"⏭️ Skipping (already done): {video_file}")
            continue

        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]

        print(f"\n🎥 Processing: {video_file}")

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_faces = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_SKIP == 0:
                faces = detector.detect_faces_from_frame(frame)

                for face in faces:
                    face = cv2.resize(face, (224, 224))
                    save_path = os.path.join(
                        output_dir,
                        f"{video_name}_f{frame_count}_{saved_faces}.jpg"
                    )
                    cv2.imwrite(save_path, face)
                    saved_faces += 1

            frame_count += 1

        cap.release()
        save_progress(progress_file, video_file)

        print(f"✅ Saved {saved_faces} faces")


if __name__ == "__main__":
    print("🚀 Phase 1.3 — Resume-Safe Face Extraction Started")
    run_extraction("real")
    run_extraction("fake")
    print("🎉 Phase 1.3 Completed Successfully")
