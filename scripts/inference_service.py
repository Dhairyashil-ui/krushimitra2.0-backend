import sys
import json
import os
import io
import contextlib
import time

# Suppress standard logs immediately
os.environ["YOLO_VERBOSE"] = "False"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import matplotlib
matplotlib.use('Agg') # Force headless backend

# Heavy imports moved to top for initialization
try:
    import cv2
    import numpy as np
    from PIL import Image
    import tensorflow as tf
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
YOLO_MODEL_PATH = "yolov8n.pt" 
CUSTOM_MODEL_PATH = os.path.join(BACKEND_ROOT, "agrimater_model2.h5")

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

# Helper function to load MobileNet
def load_custom_mobilenet():
    """Loads and configures the MobileNetV3 model in Keras."""
    try:
        log_debug(f"Loading Custom MobileNetV3 Keras model from {CUSTOM_MODEL_PATH}...")
        
        if not os.path.exists(CUSTOM_MODEL_PATH):
            log_error(f"Custom model not found at {CUSTOM_MODEL_PATH}")
            sys.exit(1)
            
        model = tf.keras.models.load_model(CUSTOM_MODEL_PATH, compile=False)
        log_debug("Custom MobileNet Keras model loaded successfully.")
        
        return model
    except Exception as e:
        log_error(f"Failed to load MobileNet: {e}")
        sys.exit(1)

# -------------------------------------------------------------
# GLOBAL MODEL LOADING (Exact User Request)
# -------------------------------------------------------------
print("🔄 Loading models once...", file=sys.stderr)

try:
    # 1. Load YOLO (SKIPPED)
    YOLO_MODEL = None

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
        "disease_analysis": {"disease_name": "Unknown", "confidence": 0.0, "model": "Local-MobileNetV3 (Keras)"}
    }
    
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
             log_error(f"Could not read image: {image_path}")
             return {"success": False, "error": "Could not read image file", "id": request_id}
             
        h, w = img.shape[:2]
        log_info(f"Original Image Size: {w}x{h}")
        
        # Skip YOLO and use full image
        log_info("YOLO Inference SKIPPED (User Request)") 
        
        # Simulate full image as the "crop"
        cropped_img_cv2 = img
        
        results_data["leaf_detection"]["objects"] = 1
        results_data["leaf_detection"]["detected"] = True # Assume leaf is present

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

    # The user mentioned Input Image Size: 160 x 160
    img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (160, 160))
    
    # Preprocess
    img_array = np.array(img_resized, dtype=np.float32)
    # Applying standard MobileNetV3 preprocessing (tf.keras.applications.mobilenet_v3.preprocess_input)
    img_preprocessed = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
    input_batch = np.expand_dims(img_preprocessed, axis=0)

    # Inference
    probabilities = model.predict(input_batch, verbose=0)[0]
    
    class_idx = np.argmax(probabilities)
    confidence = float(probabilities[class_idx])
    
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
