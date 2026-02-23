import sys
import json
import os
import time

# Suppress standard logs immediately
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Force Keras 2 legacy mode to prevent positional argument loading errors with .h5 models in Keras 3
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import matplotlib
matplotlib.use('Agg') # Force headless backend

# Heavy imports moved to top for initialization
try:
    import cv2
    import numpy as np
    import tensorflow as tf
    try:
        from tf_keras.models import Model
        from tf_keras.layers import Dense, GlobalAveragePooling2D, Dropout
        from tf_keras.applications import MobileNetV3Small
        from tf_keras.preprocessing.image import img_to_array
    except ImportError:
        # Fallback to older standard tensorflow.keras if tf_keras isn't installed
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
        from tensorflow.keras.applications import MobileNetV3Small
        from tensorflow.keras.preprocessing.image import img_to_array
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
BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOM_MODEL_PATH = os.path.join(BACKEND_ROOT, "agrimater_model2.h5")
CLASS_INDICES_PATH = os.path.join(BACKEND_ROOT, "class_indices.json")

# Global variables
MOBILENET_MODEL = None
CLASS_INDICES = {}
IDX_TO_CLASS = {}

# Helper function to load the TensorFlow model and classes
def load_custom_mobilenet():
    """Loads and configures the MobileNetV3 Keras H5 model and its classes."""
    global MOBILENET_MODEL, CLASS_INDICES, IDX_TO_CLASS
    try:
        # Load class indices
        if os.path.exists(CLASS_INDICES_PATH):
            with open(CLASS_INDICES_PATH, 'r') as f:
                CLASS_INDICES = json.load(f)
                # Invert mapping from name->idx to idx->name
                IDX_TO_CLASS = {int(v): k for k, v in CLASS_INDICES.items()}
            log_debug(f"Loaded {len(CLASS_INDICES)} classes from {CLASS_INDICES_PATH}")
        else:
            log_error(f"Class indices file not found at {CLASS_INDICES_PATH}")
            sys.exit(1)

        # Load Weights
        log_debug(f"Loading Custom MobileNetV3 H5 weights from {CUSTOM_MODEL_PATH}...")
        if os.path.exists(CUSTOM_MODEL_PATH):
            IMG_SIZE = 160
            
            base_model = MobileNetV3Small(
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
                include_top=False,
                weights=None   # We load weights manually
            )

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.3)(x)
            output = Dense(len(CLASS_INDICES), activation="softmax")(x)

            MOBILENET_MODEL = Model(inputs=base_model.input, outputs=output)
            
            # Load weights (not full model)
            MOBILENET_MODEL.load_weights(CUSTOM_MODEL_PATH)
            
            MOBILENET_MODEL.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
            
            # Test once to initialize
            dummy_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            MOBILENET_MODEL.predict(dummy_input, verbose=0)
            
            log_debug("Custom MobileNet architecture built and weights loaded successfully.")
        else:
            log_error(f"Custom model not found at {CUSTOM_MODEL_PATH}")
            sys.exit(1)
            
    except Exception as e:
        log_error(f"Failed to load MobileNet: {e}")
        sys.exit(1)

# -------------------------------------------------------------
# GLOBAL MODEL LOADING
# -------------------------------------------------------------
print("🔄 Loading models once...", file=sys.stderr)

try:
    load_custom_mobilenet()
    print("✅ Models and classes loaded successfully", file=sys.stderr)
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
        "leaf_detection": {"detected": True, "objects": 1, "model": "PlantNet-Bypass"},
        "disease_analysis": {"disease_name": "Unknown", "confidence": 0.0, "model": "TensorFlow-MobileNet-H5"}
    }
    
    # GUARD: UNSUPPORTED CROP CHECK
    plant_name = data.get("plant_name", "Unknown Plant").lower()
    supported_crops = [
        "apple", "blueberry", "cherry", "corn", "grape", "orange", 
        "peach", "pepper", "potato", "raspberry", "soybean", "squash", 
        "strawberry", "tomato"
    ]
    
    is_supported = any(crop in plant_name for crop in supported_crops)
    
    if not is_supported and plant_name != "unknown plant":
        log_info(f"Crop '{plant_name}' is not in the 38-class MobileNet dataset. Bypassing ML.")
        results_data["disease_analysis"] = {
            "disease_name": f"{plant_name.title()} - Disease models under training",
            "confidence": 1.0,
            "model": "None (Unsupported Crop)",
            "status": "Unsupported Crop"
        }
        return results_data
        
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
             log_error(f"Could not read image: {image_path}")
             return {"success": False, "error": "Could not read image file", "id": request_id}
             
        h, w = img.shape[:2]
        log_info(f"Original Image Size: {w}x{h}")
        
        # We assume the API already found a plant. No YOLO needed.

        # Custom MobileNet Classification
        mobilenet_start = time.time()
        disease_info = mobilenet_predict(MOBILENET_MODEL, img)
        log_info(f"MobileNet Inference took {time.time() - mobilenet_start:.3f}s")
        log_info(f"Total Logic Time: {time.time() - start_ts:.3f}s")
        results_data["disease_analysis"].update(disease_info)

    except Exception as e:
        log_error(f"Inference error: {e}")
        return {"success": False, "error": str(e), "id": request_id}

    return results_data

def mobilenet_predict(model, cv2_image):
    """
    Runs TensorFlow MobileNet inference on a CV2 image (numpy array).
    Returns a dictionary with disease, confidence, and raw_classification.
    """
    if cv2_image is None:
        return {"disease_name": "Error", "confidence": 0.0, "status": "No Image"}

    try:
        # Preprocessing expected by MobileNetV3/TensorFlow models
        # Target size 160x160 for our custom model
        img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (160, 160))
        
        # Convert to float and normalize if required (typical is to divide by 255.0 or keras preprocess)
        # MobileNetV3 in keras usually expects pixels in [-1, 1] or [0, 255] or [0, 1] depending on how it was built.
        # Most of the time standard division by 255.0 works for custom trained models.
        img_array = img_to_array(img_resized)
        img_array = img_array / 255.0
        input_batch = np.expand_dims(img_array, axis=0)

        # Prediction
        preds = model.predict(input_batch, verbose=0)
        probabilities = preds[0]
        
        class_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[class_idx])
        
        category_name = IDX_TO_CLASS.get(class_idx, "Unknown")

        # Format category name nicely to show crop and condition
        formatted_name = category_name.replace("___", " - ").replace("_", " ")

        # GUARD: CONFIDENCE THRESHOLD
        CONFIDENCE_THRESHOLD = 0.45 
        result = {
            "disease_name": formatted_name,
            "confidence": round(confidence, 4),
            "raw_classification": category_name
        }
        
        if confidence < CONFIDENCE_THRESHOLD:
            result["status"] = "Low Confidence"
            
        return result
    except Exception as e:
        log_error(f"Error during prediction: {e}")
        return {"disease_name": "Error", "confidence": 0.0, "status": f"Error: {e}"}

def main():
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
