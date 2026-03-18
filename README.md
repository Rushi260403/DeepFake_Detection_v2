# 🎭 DeepFake Detection System

A Deep Learning-based web application to detect whether a video is **Real or Fake (DeepFake)** using Computer Vision and Neural Networks.

---

## 🚀 Project Overview

Deepfake technology can generate highly realistic fake videos. This project detects such manipulated videos using:

- Face Detection (YOLOv8)
- Deep Learning Model (CNN)
- Frame Analysis + Voting Mechanism
- Flask Web Interface

Users can upload a video and get a prediction with confidence.

---
## Project Structure
DeepFake_Detection_v2/
│
├── backend/ # Flask backend
│ ├── app.py
│ └── video_service.py
│
├── detection/ # Face detection & prediction logic
│ ├── face_detector.py
│ ├── video_face_extractor.py
│ └── video_predictor.py
│
├── model/ # Trained models
│ ├── face_classifier_balanced.h5
│ └── yolov8n.pt
│
├── dataset/ # Dataset (training data)
│
├── evaluation/ # Evaluation scripts
│
├── scripts/ # Data processing scripts
│
├── ui/ # Frontend (User Interface)
│ ├── templates/
│ │ └── index.html
│ │
│ └── static/
│ ├── style.css
│ ├── script.js
│ └── uploads/ # Uploaded videos
│
├── utils/ # Helper functions
│ └── voting.py
│
├── config.py # Configuration file
├── requirements.txt # Dependencies
└── README.md # Project documentation
## 🧠 Features

- Upload video through web interface  
- Extract frames from video  
- Detect faces using YOLOv8  
- Classify faces using CNN model  
- Final prediction using voting  
- Output: **REAL / FAKE**  

---

## ⚙️ Technologies Used

- Python  
- TensorFlow / Keras  
- OpenCV  
- Flask  
- NumPy  
- YOLOv8  
- HTML, CSS, JavaScript  

---

## 🔄 Workflow
Upload Video → Extract Frames → Detect Faces → Classify Faces → Voting → Result
