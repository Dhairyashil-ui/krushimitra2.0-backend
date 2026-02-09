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
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
YOLO_MODEL_PATH = "yolov8n.pt"

# Global Models
yolo_model = None
mobilenet_model = None
mobilenet_preprocess = None
mobile_net_weights = None

def load_models():
    """Load models once at startup."""
    global yolo_model, mobilenet_model, mobilenet_preprocess, mobile_net_weights
    
    try:
        log_debug("Importing AI Libraries...")
        from ultralytics import YOLO
        import torch
        from torchvision import models
        
        # 1. Load YOLO
        log_debug(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
        yolo_model = YOLO(YOLO_MODEL_PATH)
        
        # 2. Load MobileNetV3
        log_debug("Loading MobileNetV3 model...")
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        mobile_net_weights = weights # store for later use (categories)
        mobilenet_model = models.mobilenet_v3_large(weights=weights)
        mobilenet_model.eval()
        
        mobilenet_preprocess = weights.transforms()
        
        log_debug("Models loaded successfully.")
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
        "disease_analysis": {"disease": "Unknown", "confidence": 0.0, "model": "MobileNetV3"}
    }
    
    try:
        # STEP 1: YOLO Detection
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
        original_img = cv2.imread(image_path)
        
        # Crop logic
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

        # STEP 2: MobileNet Classification
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
            category_name = mobile_net_weights.meta["categories"][top_catid[0].item()]
            
            results_data["disease_analysis"]["confidence"] = confidence
            results_data["disease_analysis"]["raw_classification"] = category_name
            results_data["disease_analysis"]["disease"] = f"Detected Object: {category_name}"

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
