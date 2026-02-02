import cv2
import os
from ultralytics import YOLO


class YOLOFaceDetector:
    def __init__(self):
        # YOLOv8 nano model (CPU friendly)
        self.model = YOLO("yolov8n.pt")

    def detect_faces(self, image_path, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        image = cv2.imread(image_path)
        if image is None:
            return 0

        results = self.model(image, conf=0.4)
        face_count = 0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = image[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                face = cv2.resize(face, (224, 224))
                save_path = os.path.join(save_dir, f"face_{face_count}.jpg")
                cv2.imwrite(save_path, face)
                face_count += 1

        return face_count
