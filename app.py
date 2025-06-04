from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import numpy as np
import torch
from src.anti_spoof_predict import AntiSpoofPredict
import os
import tempfile

app = Flask(__name__)
CORS(app)

# Initialize model
device_id = 0
model_path = "./resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
anti_spoof_predictor = AntiSpoofPredict(device_id)
anti_spoof_predictor._load_model(model_path)

def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']

    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, "input_video.mp4")
    video_file.save(temp_video_path)

    try:
        frames = read_video_frames(temp_video_path)
        if not frames:
            return jsonify({"error": "Could not read video frames"}), 400

        # Use middle frame
        frame = frames[len(frames) // 2]

        # Get face bounding box
        bbox = anti_spoof_predictor.get_bbox(frame)
        x, y, w, h = bbox

        # Crop and resize
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (80, 80))  # match model input size

        # Predict
        prediction = anti_spoof_predictor.predict(face_img, model_path)
        prob_real = float(prediction[0][0])
        prob_spoof = float(prediction[0][1])
        result = "real" if prob_real > prob_spoof else "spoof"

        return jsonify({
            "result": result,
            "prob_real": prob_real,
            "prob_spoof": prob_spoof
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

if __name__ == '__main__':
    app.run(debug=True)
