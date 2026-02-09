import subprocess
import json
import time
import os

# Path to the inference service
SERVICE_PATH = r"d:\finalkrushimitra_001\KrushiMitra-Backend\scripts\inference_service.py"
# Path to an image. We need a real image path. I'll search for one or use a dummy path if the service handles file-not-found gracefully 
# (it returns error, which is fine for testing the service loop).
# But for crop guard test, file existence is checked inside process_request *after* JSON parsing?
# No, actually line 102 checks existence first.
# So I need a valid image path to test the full flow, OR I can rely on the fact that "mango" guard returns early?
# Let's check the code:
# Line 101: get image_path
# Line 102: if not exists -> return error.
# The crop guard is at the very top of `process_request` in my modification?
# Let's check where I inserted it. 
# I inserted it at the start of `try` block (around line 112 in original, or line 110 after edits).
# Wait, line 102 is BEFORE the try block in original?
# Let's look at `inference_service.py` again.

# Original:
# 94: def process_request(data):
# 101: image_path = data.get("image_path")
# 102: if not image_path ... return error
# 111: try: ...

# My edit:
# I replaced from line 111.
# So crop guard is INSIDE the try block, AFTER the image path check.
# So I need a valid image file to test the "mango" guard if I want to avoid the "File not found" error first.

# I'll create a dummy image first.

def create_dummy_image():
    from PIL import Image
    img = Image.new('RGB', (100, 100), color = 'red')
    img.save("dummy_test.jpg")
    return os.path.abspath("dummy_test.jpg")

def test_service():
    img_path = create_dummy_image()
    
    # Start the service
    proc = subprocess.Popen(
        ["python", SERVICE_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Read the startup "Service Ready" log line from stderr if possible, or just wait a bit
    time.sleep(5) 
    
    # Test cases
    test_inputs = [
        {"image_path": img_path, "plant_name": "Mango"},
        {"image_path": img_path, "plant_name": "Tomato"},
        {"image_path": img_path, "plant_name": "Unknown"},
    ]
    
    for inp in test_inputs:
        print(f"Sending: {json.dumps(inp)}")
        proc.stdin.write(json.dumps(inp) + "\n")
        proc.stdin.flush()
        
        # Read response
        response_line = proc.stdout.readline()
        print(f"Received: {response_line.strip()}")
        
    proc.terminate()
    try:
        os.remove(img_path)
    except:
        pass

if __name__ == "__main__":
    test_service()
