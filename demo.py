# 🧸 ToyVision-AI: Real-time Toy Recognition Demo 
# ---------------------------------------------------------------------------------------
# Uses a webcam + Edge Impulse model to detect toys and announce them using voice.
# Bounding boxes are drawn accurately on the full-resolution webcam feed.
# Prevents repeating the same phrase more than once every 10 seconds.

import cv2
import numpy as np
import pyttsx3
from edge_impulse_linux.image import ImageImpulseRunner
import os
import time

import platform
import os

# Automatically select model file based on OS and architecture
base_path = os.path.dirname(__file__)
system = platform.system()
machine = platform.machine()

if system == "Darwin":
    model_file = "modelfile_mac.eim"
elif system == "Linux":
    if "armv7" in machine:
        model_file = "modelfile_linux-armv7.eim"
    elif "aarch64" in machine:
        model_file = "modelfile_linux-aarch64.eim"
    else:
        raise Exception(f"Unsupported platform: {system} ({machine})")
else:
    raise Exception(f"Unsupported platform: {system} ({machine})")

MODEL_PATH = os.path.join(base_path, model_file)

# Load Edge Impulse model
runner = ImageImpulseRunner(MODEL_PATH)
model_info = runner.init()
labels = model_info['model_parameters']['labels']

# Get model input details
input_width = model_info['model_parameters']['image_input_width']
input_height = model_info['model_parameters']['image_input_height']
input_channels = model_info['model_parameters']['image_channel_count']

# Init TTS engine and cooldown tracking
engine = pyttsx3.init()
last_label = None
last_spoken_time = 0
COOLDOWN_SECONDS = 10
CONFIDENCE_THRESHOLD = 0.9

# Activate webcam
cap = cv2.VideoCapture(0)
print("🎥 Camera activated — press 'q' to quit the demo.")

# Resize and crop frame for inference input (Fit Shortest Axis + Center Crop)
def resize_with_aspect_and_get_roi(frame, target_size):
    h, w, _ = frame.shape
    aspect = w / h
    if h < w:
        new_h = target_size
        new_w = int(aspect * target_size)
    else:
        new_w = target_size
        new_h = int(target_size / aspect)

    resized = cv2.resize(frame, (new_w, new_h))
    x_start = (new_w - target_size) // 2
    y_start = (new_h - target_size) // 2
    cropped = resized[y_start:y_start + target_size, x_start:x_start + target_size]

    # Scale and offset to remap bounding boxes
    scale = w / new_w if h < w else h / new_h
    x_offset = int(x_start * scale)
    y_offset = int(y_start * scale)

    return cropped, scale, x_offset, y_offset

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Could not read from camera.")
        break

    # Resize input and get mapping info
    resized_input, roi_scale, x_offset, y_offset = resize_with_aspect_and_get_roi(frame, input_width)

    # Convert BGR → RGB if needed
    if input_channels == 3:
        resized_input = cv2.cvtColor(resized_input, cv2.COLOR_BGR2RGB)

    # Run inference
    features, _ = runner.get_features_from_image(resized_input)
    res = runner.classify(features)
    print(f"📊 Raw result: {res}")

    # Draw detections on full-res frame
    if "bounding_boxes" in res["result"]:
        for bb in res["result"]["bounding_boxes"]:
            label = bb['label']
            confidence = bb['value']
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            # Remap bounding box coordinates to full-res frame
            x1 = int(bb['x'] * roi_scale) + x_offset
            y1 = int(bb['y'] * roi_scale) + y_offset
            x2 = x1 + int(bb['width'] * roi_scale)
            y2 = y1 + int(bb['height'] * roi_scale)

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Speak once per label or after cooldown
            current_time = time.time()
            if label != last_label or (current_time - last_spoken_time) > COOLDOWN_SECONDS:
                print(f"🧠 Detected: {label} (confidence: {confidence:.2f})")
                engine.say(f"That's a {label}")
                engine.runAndWait()
                last_label = label
                last_spoken_time = current_time
    else:
        last_label = None

    # Display annotated frame
    cv2.namedWindow('ToyVision AI Demo', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('ToyVision AI Demo', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('ToyVision AI Demo', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting ToyVision demo.")
        break

# Cleanup
cap.release()
runner.stop()
cv2.destroyAllWindows()
