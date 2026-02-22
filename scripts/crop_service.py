import sys
import json
import os
import logging
from flask import Flask, request, jsonify
from waitress import serve
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CropService')

# Suppress logs and GUI
os.environ["YOLO_VERBOSE"] = "False"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import matplotlib
matplotlib.use('Agg') # Force headless backend

# Define Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
# Check where the model actually is. File list showed it in root of backend.
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
if not os.path.exists(YOLO_MODEL_PATH):
    # Fallback to models dir
    YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")
    if not os.path.exists(YOLO_MODEL_PATH):
        # Fallback to just filename (let YOLO download/find it)
        YOLO_MODEL_PATH = "yolov8n.pt"

app = Flask(__name__)

# Global variables for models
yolo_model = None
mobilenet_model = None

# Import libraries lazily or at module level? 
# For a persistent service, module level is fine, but we'll load models in a function to handle errors gracefully.
try:
    logger.info("Importing ML libraries...")
    from ultralytics import YOLO
    import cv2
    from PIL import Image
    import tensorflow as tf
    logger.info("Libraries imported successfully.")
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    sys.exit(1)

# Custom Classes (38 from plant village)
CLASSES = [
    "Apple_scab", "Apple_black_rot", "Apple_cedar_apple_rust", "Apple_healthy",
    "Blueberry_healthy", "Cherry_powdery_mildew", "Cherry_healthy",
    "Corn_gray_leaf_spot", "Corn_common_rust", "Corn_northern_leaf_blight", "Corn_healthy",
    "Grape_black_rot", "Grape_black_measles", "Grape_leaf_blight", "Grape_healthy",
    "Orange_haunglongbing", "Peach_bacterial_spot", "Peach_healthy",
    "Pepper_bacterial_spot", "Pepper_healthy", "Potato_early_blight",
    "Potato_late_blight", "Potato_healthy", "Raspberry_healthy",
    "Soybean_healthy", "Squash_powdery_mildew", "Strawberry_healthy",
    "Strawberry_leaf_scorch", "Tomato_bacterial_spot", "Tomato_early_blight",
    "Tomato_late_blight", "Tomato_leaf_mold", "Tomato_septoria_leaf_spot",
    "Tomato_spider_mites", "Tomato_target_spot", "Tomato_yellow_leaf_curl_virus",
    "Tomato_mosaic_virus", "Tomato_healthy"
]

# Define Custom Model Path
CUSTOM_MODEL_PATH = os.path.join(BASE_DIR, "agrimater_model2.h5")

def load_models():
    global yolo_model, mobilenet_model
    
    try:
        # Load YOLO Model
        logger.info(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
        yolo_model = YOLO(YOLO_MODEL_PATH)
        logger.info("YOLO model loaded.")
        
        # Load Custom MobileNetV3 Model
        logger.info(f"Loading Custom MobileNetV3 Keras model from {CUSTOM_MODEL_PATH}...")
        
        if os.path.exists(CUSTOM_MODEL_PATH):
            mobilenet_model = tf.keras.models.load_model(CUSTOM_MODEL_PATH, compile=False)
            logger.info("Custom MobileNet Keras model loaded successfully.")
        else:
            logger.error(f"Custom model not found at {CUSTOM_MODEL_PATH}")
            return False

        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    if yolo_model and mobilenet_model:
        return jsonify({"status": "healthy", "models_loaded": True}), 200
    else:
        return jsonify({"status": "unhealthy", "models_loaded": False}), 503

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if not request.json or 'image_path' not in request.json:
        return jsonify({"success": False, "error": "Missing image_path"}), 400
    
    image_path = request.json['image_path']
    
    if not os.path.exists(image_path):
        return jsonify({"success": False, "error": "File not found", "details": f"Path: {image_path}"}), 404

    results_data = {
        "success": True,
        "leaf_detection": {
            "detected": False,
            "objects": 0,
            "model": "YOLOv8n"
        },
        "disease_analysis": {
            "disease": "Unknown",
            "confidence": 0.0,
            "model": "MobileNetV3 (Keras)"
        }
    }

    cropped_img_cv2 = None

    # Step 1: YOLOv8 Leaf Detection & Cropping
    try:
        detections = yolo_model(image_path)
        # logger.info(f"YOLO inference complete. Found {len(detections)} detection result objects.")
        
        best_box = None
        max_area = 0
        detected_objects = []

        if len(detections) > 0:
            for box in detections[0].boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                xyxy = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]
                area = w * h

                detected_objects.append({
                    "class": cls,
                    "conf": conf,
                    "box": box.xywh.tolist()[0]
                })

                if area > max_area:
                    max_area = area
                    best_box = xyxy
        
        results_data["leaf_detection"]["objects"] = len(detected_objects)
        
        original_img = cv2.imread(image_path)
        if original_img is None:
             raise ValueError("Failed to read image with cv2")

        if best_box is not None:
            results_data["leaf_detection"]["detected"] = True
            x1, y1, x2, y2 = map(int, best_box)
            h_img, w_img = original_img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            if x2 > x1 and y2 > y1:
                cropped_img_cv2 = original_img[y1:y2, x1:x2]
            else:
                cropped_img_cv2 = original_img
        else:
            cropped_img_cv2 = original_img

    except Exception as e:
        logger.error(f"YOLO/CV2 Error: {e}")
        results_data["leaf_detection"]["error"] = str(e)
        # Attempt to load image if cropping failed but image exists
        if cropped_img_cv2 is None:
             cropped_img_cv2 = cv2.imread(image_path)

    # Step 2: MobileNetV3 Classification
    if cropped_img_cv2 is not None:
        try:
            # Resize appropriately
            img_rgb = cv2.cvtColor(cropped_img_cv2, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (160, 160))

            # Preprocess
            img_array = np.array(img_resized, dtype=np.float32)
            img_preprocessed = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
            input_batch = np.expand_dims(img_preprocessed, axis=0)

            # Inference
            probabilities = mobilenet_model.predict(input_batch, verbose=0)[0]
            
            class_idx = np.argmax(probabilities)
            confidence = float(probabilities[class_idx])
            
            if 0 <= class_idx < len(CLASSES):
                category_name = CLASSES[class_idx]
            else:
                category_name = "Unknown"

            results_data["disease_analysis"]["confidence"] = confidence
            results_data["disease_analysis"]["raw_classification"] = category_name
            results_data["disease_analysis"]["disease"] = category_name
            
        except Exception as e:
            logger.error(f"MobileNet Error: {e}")
            results_data["disease_analysis"]["error"] = str(e)
            results_data["disease_analysis"]["disease"] = "Error"

    return jsonify(results_data)

if __name__ == "__main__":
    if load_models():
        logger.info("Starting Crop Disease Analysis Service on port 5002...")
        # Use Waitress for production-ready serving
        serve(app, host='0.0.0.0', port=5002, threads=4)
    else:
        logger.error("Failed to load models. Exiting.")
        sys.exit(1)
