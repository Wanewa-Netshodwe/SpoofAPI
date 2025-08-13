from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import cv2
import numpy as np
import torch
from src.anti_spoof_predict import AntiSpoofPredict
import os
import tempfile
from deepface import DeepFace
import base64
from io import BytesIO
from PIL import Image
import subprocess

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


def base64_to_image(base64_str):
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    return np.array(img)


def convert_to_mp4(input_path, output_path):
    try:
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-strict", "experimental",
            output_path
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print("FFmpeg conversion error:", e)
        return False


@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    student_number = request.form.get('student_number') or (request.json and request.json.get('student_number'))
    if not student_number:
        return jsonify({"error": "student_number is required"}), 400

    temp_dir = tempfile.mkdtemp()
    temp_webm_path = os.path.join(temp_dir, "input_video.webm")
    temp_mp4_path = os.path.join(temp_dir, "input_video.mp4")

    video_file.save(temp_webm_path)

    # Convert WebM to MP4
    if not convert_to_mp4(temp_webm_path, temp_mp4_path):
        return jsonify({"error": "Failed to convert video"}), 500

    try:
        frames = read_video_frames(temp_mp4_path)
        if not frames:
            return jsonify({"error": "Could not read video frames"}), 400

        # Use middle frame for face cropping
        frame = frames[len(frames) // 2]
        bbox = anti_spoof_predictor.get_bbox(frame)
        x, y, w, h = bbox

        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (80, 80))

        prediction = anti_spoof_predictor.predict(face_img, model_path)
        prob_real = float(prediction[0][0])
        prob_spoof = float(prediction[0][1])
        result = "real" if prob_real > prob_spoof else "spoof"

        if result == "spoof":
            NODE_URL = "http://localhost:3001/api/faceRec/employee_face"
            headers = {"secret": "da_SeCret"}
            payload = {"student_number": student_number}
            node_response = requests.post(NODE_URL, json=payload, headers=headers)

            if node_response.status_code == 201:
                data = node_response.json()
                node_face_base64 = data.get("image_base64")
                if not node_face_base64:
                    return jsonify({"error": "Node API did not return face image"}), 500

                node_face_img = base64_to_image(node_face_base64)
                node_face_img = cv2.resize(node_face_img, (80, 80))

                # Convert BGR to RGB for DeepFace
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                node_face_img_rgb = cv2.cvtColor(node_face_img, cv2.COLOR_BGR2RGB)

                try:
                    verification = DeepFace.verify(face_img_rgb, node_face_img_rgb, enforce_detection=False)
                    return jsonify({
                        "result": result,
                        "prob_real": prob_real,
                        "prob_spoof": prob_spoof,
                        "node_response": data,
                        "face_match": verification['verified'],
                        "distance": verification['distance']
                    })
                except Exception as e:
                    return jsonify({"error": f"DeepFace verification failed: {str(e)}"}), 500
            else:
                return jsonify({
                    "error": "Node API call failed",
                    "status_code": node_response.status_code,
                    "response": node_response.text
                }), 500
        else:
            return jsonify({
                "result": result,
                "prob_real": prob_real,
                "prob_spoof": prob_spoof
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temporary files
        for f in [temp_webm_path, temp_mp4_path]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


if __name__ == '__main__':
    
    app.run(debug=True)
