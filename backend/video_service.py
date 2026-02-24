import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(PROJECT_ROOT)

from detection.video_predictor import predict_video


def analyze_video(video_path):

    result = predict_video(video_path)

    prediction = result["prediction"]
    confidence = float(result["confidence"])
    frames = result["frames"]

    return prediction, confidence, frames