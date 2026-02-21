import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(PROJECT_ROOT)

from detection.video_predictor import predict_video

def analyze_video(video_path):

    label, confidence, frames = predict_video(video_path)

    frame_urls = []

    for f in frames[:10]:
        name = f.split("\\")[-1]
        frame_urls.append("/frame_results/" + name)

    return label, confidence, frame_urls