import os
import cv2
from face_detector import YOLOFaceDetector

# ===============================
# PROJECT ROOT (VERY IMPORTANT)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
VIDEO_DIR = os.path.join(DATASET_DIR, "videos")
FRAME_DIR = os.path.join(DATASET_DIR, "frames")

FRAME_SKIP = 30   # process 1 frame per second (CPU friendly)

detector = YOLOFaceDetector()


def run_extraction(class_name):
    video_dir = os.path.join(VIDEO_DIR, class_name)
    output_dir = os.path.join(FRAME_DIR, class_name)

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(video_dir):
        print(f"❌ Video directory not found: {video_dir}")
        return

    print(f"\n📂 Processing class: {class_name}")

    for video_file in os.listdir(video_dir):
        if not video_file.lower().endswith((".mp4", ".avi", ".mov")):
            continue

        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_faces = 0

        print(f"🎥 Processing video: {video_file}")

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
        print(f"✅ Saved {saved_faces} faces from {video_file}")


if __name__ == "__main__":
    print("🚀 Starting FACE EXTRACTION (YOLO)")
    run_extraction("real")
    run_extraction("fake")
    print("🎉 Phase 1.2 completed")
