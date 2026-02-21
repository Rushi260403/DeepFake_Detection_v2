import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from detection.video_predictor import predict_video


def analyze_video(video_path):

    # Call predictor (returns dictionary)
    result = predict_video(video_path)

    prediction = result["prediction"]
    confidence = result["confidence"]
    frames_data = result["frames"]

    frame_urls = []

    # Extract only paths for frontend
    for frame in frames_data[:10]:

        frame_urls.append(frame["path"])

    return prediction, confidence, frame_urls