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
    # Remove base64 header if present
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]

    try:
        img_data = base64.b64decode(base64_str)
        img = Image.open(BytesIO(img_data))
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}")


@app.route('/predict', methods=['POST'])
def predict():
    print("Current working directory:", os.getcwd())
    print("received a request")

    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    student_number = request.form.get('student_number') or (request.json and request.json.get('student_number'))
    if not student_number:
        return jsonify({"error": "student_number is required"}), 400

    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, "input_video.mp4")
    video_file.save(temp_video_path)

    try:
        frames = read_video_frames(temp_video_path)
        if not frames:
            return jsonify({"error": "Could not read video frames"}), 400

        # Use middle frame for face cropping
        frame = frames[len(frames) // 2]
        bbox = anti_spoof_predictor.get_bbox(frame)
        x, y, w, h = bbox

        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (80, 80))
        cv2.imwrite("face_crop.jpg", face_img)  # optional debugging

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
                # Assumed base64 image returned by node API in 'face_image_base64' key
                node_face_base64 = data.get("image_base64")
                if not node_face_base64:
                    return jsonify({"error": "Node API did not return face image"}), 500
                
                node_face_img = base64_to_image(node_face_base64)
                node_face_img = cv2.resize(node_face_img, (80, 80))
                cv2.imwrite("node_face.jpg", node_face_img)  # optional debugging

                # Convert BGR (OpenCV) to RGB for DeepFace
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                node_face_img_rgb = cv2.cvtColor(node_face_img, cv2.COLOR_BGR2RGB)

                try:
                    verification = DeepFace.verify(face_img_rgb, node_face_img_rgb, enforce_detection=False)
                    # verification is a dict: {'verified': bool, 'distance': float, ...}

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
            # If real, no need to call Node API or DeepFace
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
