import cv2
import os
import time

SAVE_DIR = "data/uncertain"
UNCERTAINTY_THRESHOLD = 0.55

os.makedirs(SAVE_DIR, exist_ok=True)


def is_uncertain(confidence):
    """Check if a detection's confidence is below the uncertainty threshold."""
    return confidence < UNCERTAINTY_THRESHOLD


def save_uncertain_frame(frame, confidence=None):
    """Save a frame that contains uncertain detections for later labeling."""
    timestamp = int(time.time() * 1000)
    filename = os.path.join(SAVE_DIR, f"uncertain_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    msg = f"Saved uncertain sample: {filename}"
    if confidence is not None:
        msg += f" (conf={confidence:.3f})"
    print(msg)
