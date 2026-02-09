import sys
import json
import os
import io
import contextlib

# Suppress standard logs immediately
os.environ["YOLO_VERBOSE"] = "False"
import matplotlib
matplotlib.use('Agg') # Force headless backend

# Use stderr for all logs so stdout is clean for JSON communication
def log_debug(msg):
    sys.stderr.write(f"DEBUG: {msg}\n")
    sys.stderr.flush()

def log_error(msg):
    sys.stderr.write(f"ERROR: {msg}\n")
    sys.stderr.flush()

# Define Paths
MODELS_DIR = os.path.dirname(os.path.abspath(__file__)) # Use script dir as base if needed, or current
# We assume the script is in scripts/ and models are in ../ or root. 
# User said model is in D:\finalkrushimitra_001\KrushiMitra-Backend\leaf_disease_mobilenet.pth
# which is the parent of 'scripts/'
BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_MODEL_PATH = "yolov8n.pt" # Assuming YOLO is also in root or reachable
CUSTOM_MODEL_PATH = os.path.join(BACKEND_ROOT, "leaf_disease_mobilenet.pth")

# Custom Classes
CLASSES = [
    "Tomato - Healthy",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold"
]

# Global Models
yolo_model = None
mobilenet_model = None
mobilenet_preprocess = None

def load_models():
    """Load models once at startup."""
    global yolo_model, mobilenet_model, mobilenet_preprocess
    
    try:
        log_debug("Importing AI Libraries...")
        from ultralytics import YOLO
        import torch
        from torch import nn
        from torchvision import models
        
        # 1. Load YOLO
        # Check if yolov8n.pt exists in root, otherwise try to download or use default
        yolo_path = os.path.join(BACKEND_ROOT, YOLO_MODEL_PATH)
        if not os.path.exists(yolo_path):
             yolo_path = YOLO_MODEL_PATH # fallback to let ultralytics find/download it
             
        log_debug(f"Loading YOLO model from {yolo_path}...")
        yolo_model = YOLO(yolo_path)
        
        # 2. Load Custom MobileNetV3
        log_debug(f"Loading Custom MobileNetV3 model from {CUSTOM_MODEL_PATH}...")
        
        # Initialize model structure
        mobilenet_model = models.mobilenet_v3_large(pretrained=False)
        
        # Modify classifier head to match training (4 classes)
        # MobileNetV3 Large classifier: 
        # (3): Linear(in_features=1280, out_features=1000, bias=True) -> Change to 4
        mobilenet_model.classifier[3] = nn.Linear(1280, len(CLASSES))
        
        # Load Weights
        if os.path.exists(CUSTOM_MODEL_PATH):
            state_dict = torch.load(CUSTOM_MODEL_PATH, map_location="cpu")
            mobilenet_model.load_state_dict(state_dict)
            mobilenet_model.eval()
            log_debug("Custom MobileNet loaded successfully.")
        else:
            log_error(f"Custom model not found at {CUSTOM_MODEL_PATH}")
            # Fallback or exit? For now, we exit as this is critical
            sys.exit(1)

        # Standard MobileNet Preprocessing
        # We can use the default transforms from weights even if we don't use the weights
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        mobilenet_preprocess = weights.transforms()
        
    except Exception as e:
        log_error(f"Failed to load models: {e}")
        sys.exit(1)

def process_request(data):
    """Process a single inference request."""
    import cv2
    import torch
    from PIL import Image
    import numpy as np

    image_path = data.get("image_path")
    if not image_path or not os.path.exists(image_path):
        return {"success": False, "error": "Image file not found"}

    results_data = {
        "success": True,
        "leaf_detection": {"detected": False, "objects": 0, "model": "YOLOv8n"},
        "disease_analysis": {"disease": "Unknown", "confidence": 0.0, "model": "Local-MobileNetV3"}
    }
    
    try:
        # ---------------------------------------------------------
        # GUARD 1: CROP MODEL SELECTION
        # ---------------------------------------------------------
        plant_name = data.get("plant_name", "tomato").lower()  # Default to tomato if unspecified
        
        # Valid supported crops for this MVP
        if "mango" in plant_name:
            return {
                "success": True,
                "leaf_detection": {"detected": False, "objects": 0, "model": "None"},
                "disease_analysis": {
                    "disease": "Model under training",
                    "confidence": 1.0,
                    "model": "Future-Mango-Net"
                }
            }

        # If not tomato and not mango, generic fallback (treat as tomato for now or error?)
        # User said "Tomato -> disease". We'll assume anything else is processed as Tomato 
        # UNLESS we want to be strict. Let's stick to processing as Tomato for "unknown" to be safe/lenient,
        # but maybe log it.
        
        # STEP 1: YOLO Detection (Leaf presence)
        detections = yolo_model(image_path)
        
        best_box = None
        max_area = 0
        detected_objects = []

        if len(detections) > 0:
            for box in detections[0].boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                xyxy = box.xyxy[0].cpu().numpy() 
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
        
        # Only proceed to disease analysis if we found something, 
        # OR if we want to force run on the whole image?
        # Let's crop if found, else use whole image.
        
        original_img = cv2.imread(image_path)
        if original_img is None:
             return {"success": False, "error": "Could not read image file"}

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

        # STEP 2: Custom MobileNet Classification (Tomato)
        if cropped_img_cv2 is not None:
            img_rgb = cv2.cvtColor(cropped_img_cv2, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            input_tensor = mobilenet_preprocess(pil_img)
            input_batch = input_tensor.unsqueeze(0)

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

            # ---------------------------------------------------------
            # GUARD 2: CONFIDENCE THRESHOLD
            # ---------------------------------------------------------
            CONFIDENCE_THRESHOLD = 0.45 # Tunable
            
            if confidence < CONFIDENCE_THRESHOLD:
                category_name = "Uncertain / Healthy?" # Soft fallback
                # Or keep the name but flag it? 
                # User said "Add confidence guard".
                # Let's append status
                results_data["disease_analysis"]["status"] = "Low Confidence"
            
            results_data["disease_analysis"]["confidence"] = confidence
            results_data["disease_analysis"]["raw_classification"] = category_name
            results_data["disease_analysis"]["disease"] = category_name


    except Exception as e:
        log_error(f"Inference error: {e}")
        return {"success": False, "error": str(e)}

    return results_data

def main():
    # Load models initially
    load_models()
    
    log_debug("Service Ready. Waiting for input...")
    
    # Main Loop
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break # EOF
            
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                response = process_request(request)
            except json.JSONDecodeError:
                response = {"success": False, "error": "Invalid JSON input"}
            
            # Print response as a single JSON line to stdout
            print(json.dumps(response))
            sys.stdout.flush()
            
        except Exception as e:
            log_error(f"Loop error: {e}")
            # Try to send error response if possible
            print(json.dumps({"success": False, "error": "Internal Service Error"}))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
