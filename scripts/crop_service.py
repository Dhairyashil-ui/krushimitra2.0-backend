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
mobilenet_weights = None
mobilenet_preprocess = None

# Import libraries lazily or at module level? 
# For a persistent service, module level is fine, but we'll load models in a function to handle errors gracefully.
try:
    logger.info("Importing ML libraries...")
    from ultralytics import YOLO
    import cv2
    import torch
    from torchvision import models, transforms
    from PIL import Image
    logger.info("Libraries imported successfully.")
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    sys.exit(1)

# Custom Classes
CLASSES = [
    "Tomato - Healthy",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold"
]

# Define Custom Model Path
CUSTOM_MODEL_PATH = os.path.join(BASE_DIR, "leaf_disease_mobilenet.pth")

def load_models():
    global yolo_model, mobilenet_model, mobilenet_weights, mobilenet_preprocess
    
    try:
        # Load YOLO Model
        logger.info(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
        yolo_model = YOLO(YOLO_MODEL_PATH)
        logger.info("YOLO model loaded.")
        
        # Load Custom MobileNetV3 Model
        logger.info(f"Loading Custom MobileNetV3 model from {CUSTOM_MODEL_PATH}...")
        
        # Initialize model structure
        mobilenet_model = models.mobilenet_v3_large(weights=None) # No pretrained weights initially
        
        # Modify classifier head to match training (4 classes)
        from torch import nn
        mobilenet_model.classifier[3] = nn.Linear(1280, len(CLASSES))
        
        # Load Weights
        if os.path.exists(CUSTOM_MODEL_PATH):
            state_dict = torch.load(CUSTOM_MODEL_PATH, map_location="cpu")
            mobilenet_model.load_state_dict(state_dict)
            mobilenet_model.eval()
            logger.info("Custom MobileNet loaded successfully.")
        else:
            logger.error(f"Custom model not found at {CUSTOM_MODEL_PATH}")
            return False

        # Standard MobileNet Preprocessing
        # We can use the default transforms from the weights class even if we don't load the weights
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        mobilenet_preprocess = weights.transforms()
        mobilenet_weights = weights # Keep for metadata if needed, though we use custom classes
        
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
            "model": "MobileNetV3 (Pretrained ImageNet)"
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
            # Convert CV2 (BGR) to PIL (RGB)
            img_rgb = cv2.cvtColor(cropped_img_cv2, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # Transform
            input_tensor = mobilenet_preprocess(pil_img)
            input_batch = input_tensor.unsqueeze(0)

            # Inference
            with torch.no_grad():
                output = mobilenet_model(input_batch)
            
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 1)
            
            confidence = top_prob[0].item()
            class_idx = top_catid[0].item()
            
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
