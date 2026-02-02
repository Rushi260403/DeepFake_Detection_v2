from face_detector import YOLOFaceDetector

detector = YOLOFaceDetector()

faces = detector.detect_faces(
    image_path="test_images/test.jpg",
    save_dir="test_faces"
)

print(f"Detected faces: {faces}")
