import sys
import json
import os
import argparse
import logging
import numpy as np

# Suppress logs
os.environ["YOLO_VERBOSE"] = "False"

# Define Paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
YOLO_MODEL_PATH = "yolov8n.pt" 

def error_exit(message, details=""):
    print(json.dumps({
        "success": False,
        "error": message,
        "details": details
    }))
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("plant_name", nargs="?", default="Unknown", help="Name of the plant (optional)")
    args = parser.parse_args()

    image_path = args.image_path

    if not os.path.exists(image_path):
        error_exit("File not found", f"Image path: {image_path}")

    # ---------------------------------------------------------
    # IMPORT LIBRARIES
    # ---------------------------------------------------------
    try:
        from ultralytics import YOLO
        import cv2
        import torch
        from torchvision import models, transforms
        from PIL import Image
    except ImportError as e:
        error_exit("Missing dependencies", f"Please install torch torchvision ultralytics opencv-python: {str(e)}")

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

    # ---------------------------------------------------------
    # STEP 1: YOLOv8 LEAF DETECTION & CROPPING
    # ---------------------------------------------------------
    cropped_img_cv2 = None
    
    try:
        # Load YOLO Model
        yolo_model = YOLO(YOLO_MODEL_PATH)
        
        # Run Inference
        detections = yolo_model(image_path)
        
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

                # Select largest object as the 'subject'
                if area > max_area:
                    max_area = area
                    best_box = xyxy
        
        results_data["leaf_detection"]["objects"] = len(detected_objects)

        original_img = cv2.imread(image_path)

        if best_box is not None:
            results_data["leaf_detection"]["detected"] = True
            
            # Crop the image
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
        results_data["leaf_detection"]["error"] = str(e)
        cropped_img_cv2 = cv2.imread(image_path) if 'cv2' in locals() else None

    # ---------------------------------------------------------
    # STEP 2: MOBILENETV3 (PRETRAINED) CLASSIFICATION
    # ---------------------------------------------------------
    if cropped_img_cv2 is not None:
        try:
            # Prepare Model
            # Using MobileNetV3 Large with ImageNet weights
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
            model = models.mobilenet_v3_large(weights=weights)
            model.eval()

            # Prepare Preprocessing
            preprocess = weights.transforms()

            # Convert CV2 (BGR) to PIL (RGB)
            img_rgb = cv2.cvtColor(cropped_img_cv2, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # Transform
            input_tensor = preprocess(pil_img)
            input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

            # Inference
            with torch.no_grad():
                output = model(input_batch)
            
            # Get probabilities
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

            # Get Top 1
            top_prob, top_catid = torch.topk(probabilities, 1)
            
            confidence = top_prob[0].item()
            category_name = weights.meta["categories"][top_catid[0].item()]

            # ---------------------------------------------------------
            # SIMULATED INTERPRETATION FOR DEMO
            # ---------------------------------------------------------
            # Since ImageNet classes are things like "daisy", "pot", etc.,
            # we will return the actual detected object but map valid plant-like objects to "Healthy/Unknown"
            # so the user doesn't just see "Greenhouse" as a disease.
            
            results_data["disease_analysis"]["confidence"] = confidence
            results_data["disease_analysis"]["raw_classification"] = category_name
            
            # Heuristic: If it looks like a plant, but we don't know the disease (because model is generic),
            # we will say "Potential Issue (Requires Expert)" or "Healthy"
            
            results_data["disease_analysis"]["disease"] = f"Detected Object: {category_name}"
            
        except Exception as e:
             results_data["disease_analysis"]["error"] = str(e)
             results_data["disease_analysis"]["disease"] = "Error"

    # ---------------------------------------------------------
    # OUTPUT JSON
    # ---------------------------------------------------------
    print(json.dumps(results_data))

if __name__ == "__main__":
    main()
