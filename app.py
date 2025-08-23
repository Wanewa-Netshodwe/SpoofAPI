from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import cv2
import numpy as np
import os
import tempfile
from deepface import DeepFace
import base64
from io import BytesIO
from PIL import Image
import subprocess
import traceback

app = Flask(__name__)
CORS(app)


def read_video_frames(video_path):
    """Extract all frames from a video."""
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
    """Convert base64 string to numpy image."""
    try:
        if "base64," in base64_str:
            base64_str = base64_str.split("base64,")[1]
        img_data = base64.b64decode(base64_str)
        img = Image.open(BytesIO(img_data))
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {str(e)}")


def convert_to_mp4(input_path, output_path):
    """Convert video to mp4 using ffmpeg."""
    try:
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-strict", "experimental",
            output_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print("FFmpeg conversion error:", e.stderr.decode())
        return False


@app.route('/api/predict', methods=['POST'])
def predict():
    temp_dir = None
    try:
        # Check input
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files['video']
        student_number = request.form.get('student_number') or (request.json and request.json.get('student_number'))
        if not student_number:
            return jsonify({"error": "student_number is required"}), 400

        # Prepare temp files
        temp_dir = tempfile.mkdtemp()
        temp_webm_path = os.path.join(temp_dir, "input_video.webm")
        temp_mp4_path = os.path.join(temp_dir, "input_video.mp4")
        video_file.save(temp_webm_path)

        # Convert to mp4
        if not convert_to_mp4(temp_webm_path, temp_mp4_path):
            return jsonify({"error": "Failed to convert video"}), 500

        # Extract frames
        frames = read_video_frames(temp_mp4_path)
        if not frames:
            return jsonify({"error": "Could not read video frames"}), 400

        # Use middle frame
        frame = frames[len(frames) // 2]

        # Call Node API for reference face
        NODE_URL = "http://localhost:3001/api/faceRec/employee_face"
        headers = {"secret": "da_SeCret"}
        payload = {"student_number": student_number}
        node_response = requests.post(NODE_URL, json=payload, headers=headers)

        if node_response.status_code != 201:
            return jsonify({
                "error": "Node API call failed",
                "status_code": node_response.status_code,
                "response": node_response.text
            }), 500

        data = node_response.json()
        node_face_base64 = data.get("image_base64")
        if not node_face_base64:
            return jsonify({"error": "Node API did not return face image"}), 500

        # Decode face from Node
        node_face_img = base64_to_image(node_face_base64)

        # Convert to RGB for DeepFace
        face_img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        node_face_img_rgb = cv2.cvtColor(node_face_img, cv2.COLOR_BGR2RGB)

        # DeepFace verification
        verification = DeepFace.verify(face_img_rgb, node_face_img_rgb, enforce_detection=False)

        return jsonify({
            "result": "verified" if verification['verified'] else "not_verified",
            "distance": verification['distance'],
             "face_match": verification['verified'],
            "node_response": data
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        # Cleanup
        if temp_dir and os.path.exists(temp_dir):
            for f in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, f))
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
