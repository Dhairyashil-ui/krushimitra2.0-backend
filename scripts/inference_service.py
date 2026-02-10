import sys
import json
import os
import io
import contextlib
import time
import os

# Suppress standard logs immediately
os.environ["YOLO_VERBOSE"] = "False"
import matplotlib
matplotlib.use('Agg') # Force headless backend

# Heavy imports moved to top for initialization
try:
    import cv2
    import torch
    from torch import nn
    from torchvision import models
    from PIL import Image
    import numpy as np
    import cv2
    import torch
    from torch import nn
    from torchvision import models
    from PIL import Image
    import numpy as np
    # from ultralytics import YOLO # Removed for performance
except ImportError as e:
    sys.stderr.write(f"ERROR: Missing AI dependencies: {e}\n")
    sys.exit(1)

# Use stderr for all logs so stdout is clean for JSON communication
def log_debug(msg):
    sys.stderr.write(f"DEBUG: {msg}\n")
    sys.stderr.flush()

def log_error(msg):
    sys.stderr.write(f"ERROR: {msg}\n")
    sys.stderr.flush()

def log_info(msg):
    sys.stderr.write(f"INFO: {msg}\n")
    sys.stderr.flush()

# Define Paths
MODELS_DIR = os.path.dirname(os.path.abspath(__file__)) 
BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.dirname(os.path.abspath(__file__)) 
BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# YOLO_MODEL_PATH = "yolov8n.pt" # Removed
CUSTOM_MODEL_PATH = os.path.join(BACKEND_ROOT, "leaf_disease_mobilenet.pth") 
CUSTOM_MODEL_PATH = os.path.join(BACKEND_ROOT, "leaf_disease_mobilenet.pth")

# Custom Classes
CLASSES = [
    "Tomato - Healthy",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold"
]

# Global Preprocess placeholder
mobilenet_preprocess = None

# Helper function to load MobileNet
def load_custom_mobilenet():
    """Loads and configures the MobileNetV3 model."""
    global mobilenet_preprocess
    try:
        log_debug(f"Loading Custom MobileNetV3 model from {CUSTOM_MODEL_PATH}...")
        
        # Initialize model structure
        model = models.mobilenet_v3_large(pretrained=False)
        
        # Modify classifier head to match training (4 classes)
        model.classifier[3] = nn.Linear(1280, len(CLASSES))
        
        # Load Weights
        if os.path.exists(CUSTOM_MODEL_PATH):
            state_dict = torch.load(CUSTOM_MODEL_PATH, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            log_debug("Custom MobileNet loaded successfully.")
        else:
            log_error(f"Custom model not found at {CUSTOM_MODEL_PATH}")
            sys.exit(1)

        # Standard MobileNet Preprocessing
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        mobilenet_preprocess = weights.transforms()
        
        return model
    except Exception as e:
        log_error(f"Failed to load MobileNet: {e}")
        sys.exit(1)

# -------------------------------------------------------------
# GLOBAL MODEL LOADING (Exact User Request)
# -------------------------------------------------------------
print("🔄 Loading models once...", file=sys.stderr)

try:
    # 1. Load YOLO (Removed for performance)
    # yolo_full_path = os.path.join(BACKEND_ROOT, YOLO_MODEL_PATH)
    # if not os.path.exists(yolo_full_path):
    #      yolo_full_path = YOLO_MODEL_PATH 
         
    # log_debug(f"Loading YOLO model from {yolo_full_path}...")
    # YOLO_MODEL = YOLO(yolo_full_path)
    log_info("Skipping YOLO model loading (Optimization)")

    # 2. Load MobileNet
    MOBILENET_MODEL = load_custom_mobilenet()
    
    # 3. Mark as loaded
    print("✅ Models loaded successfully", file=sys.stderr)

except Exception as e:
    log_error(f"Critical Error Loading Models: {e}")
    sys.exit(1)


# -------------------------------------------------------------
# REQUEST HANDLER
# -------------------------------------------------------------
def process_request(data):
    """Process a single inference request using pre-loaded models."""
    image_path = data.get("image_path")
    request_id = data.get("id") 
    log_info(f"Starting processing for ID: {request_id}")
    start_ts = time.time() 
    
    if not image_path or not os.path.exists(image_path):
        return {"success": False, "error": "Image file not found", "id": request_id}

    results_data = {
        "id": request_id, 
        "success": True,
        "leaf_detection": {"detected": False, "objects": 0, "model": "YOLOv8n"},
        "disease_analysis": {"disease_name": "Unknown", "confidence": 0.0, "model": "Local-MobileNetV3"}
    }
    
    try:
        # GUARD: CROP MODEL SELECTION
        plant_name = data.get("plant_name", "tomato").lower()
        
        if "mango" in plant_name:
            return {
                "success": True,
                "id": request_id,
                "leaf_detection": {"detected": False, "objects": 0, "model": "None"},
                "disease_analysis": {
                    "disease_name": "Model under training",
                    "confidence": 1.0,
                    "model": "Future-Mango-Net"
                }
            }

        # STEP 1: YOLO Detection (SKIPPED)
        # We just read the image and optionally resize it for basic sanity, 
        # but mostly we rely on MobileNet preprocessing.
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
             log_error(f"Could not read image: {image_path}")
             return {"success": False, "error": "Could not read image file", "id": request_id}
             
        h, w = img.shape[:2]
        log_info(f"Original Image Size: {w}x{h}")

        # Basic Resize to avoid massive images in memory/logs
        # (MobileNet will resize again internally via transform, but this helps PIL)
        INPUT_SIZE = 800 
        best_box = None
        
        if w > INPUT_SIZE or h > INPUT_SIZE:
             scale = min(INPUT_SIZE/w, INPUT_SIZE/h)
             new_w, new_h = int(w*scale), int(h*scale)
             original_img = cv2.resize(img, (new_w, new_h))
             log_info(f"Resized input to: {new_w}x{new_h}")
        else:
             original_img = img

        # Mark detection as skipped/assumed
        results_data["leaf_detection"]["detected"] = True 
        results_data["leaf_detection"]["objects"] = 1
        results_data["leaf_detection"]["model"] = "Skipped (Assumption)"

        cropped_img_cv2 = original_img

        # STEP 2: Custom MobileNet Classification
        mobilenet_start = time.time()
        disease_info = mobilenet_predict(MOBILENET_MODEL, cropped_img_cv2)
        log_info(f"MobileNet Inference took {time.time() - mobilenet_start:.3f}s")
        log_info(f"Total Logic Time: {time.time() - start_ts:.3f}s")
        results_data["disease_analysis"].update(disease_info)

    except Exception as e:
        log_error(f"Inference error: {e}")
        return {"success": False, "error": str(e), "id": request_id}

    return results_data

def mobilenet_predict(model, cv2_image):
    """
    Runs MobileNet inference on a CV2 image (numpy array).
    Returns a dictionary with disease, confidence, and raw_classification.
    """
    if cv2_image is None:
        return {"disease_name": "Error", "confidence": 0.0, "status": "No Image"}

    img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Global preprocess
    input_tensor = mobilenet_preprocess(pil_img)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    
    confidence = top_prob[0].item()
    class_idx = top_catid[0].item()
    
    if 0 <= class_idx < len(CLASSES):
        category_name = CLASSES[class_idx]
    else:
        category_name = "Unknown"

    # GUARD: CONFIDENCE THRESHOLD
    CONFIDENCE_THRESHOLD = 0.45 
    result = {
        "disease_name": category_name,
        "confidence": confidence,
        "raw_classification": category_name
    }
    
    if confidence < CONFIDENCE_THRESHOLD:
        result["status"] = "Low Confidence"
        
    return result

def main():
    # Models are ALREADY LOADED globally.
    log_debug("Service Ready. Waiting for input...")
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break 
            
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                response = process_request(request)
            except json.JSONDecodeError:
                response = {"success": False, "error": "Invalid JSON input"}
            
            # Print response
            print(json.dumps(response))
            sys.stdout.flush()
            
        except Exception as e:
            log_error(f"Loop error: {e}")
            print(json.dumps({"success": False, "error": "Internal Service Error"}))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
