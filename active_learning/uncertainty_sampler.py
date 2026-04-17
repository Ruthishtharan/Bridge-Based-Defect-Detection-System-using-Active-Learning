import cv2
import os
import time
import json

SAVE_DIR            = "data/uncertain"
LOG_FILE            = "data/uncertain/log.json"
UNCERTAINTY_THRESHOLD = 0.55   # frames with conf < this are "uncertain"

os.makedirs(SAVE_DIR, exist_ok=True)


def is_uncertain(confidence: float) -> bool:
    """Return True if detection confidence is below the uncertainty threshold."""
    return confidence < UNCERTAINTY_THRESHOLD


def _load_log() -> list:
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"[Warning] log.json is corrupted ({e}). Starting with empty log.")
            return []
    return []


def _save_log(entries: list):
    with open(LOG_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def save_uncertain_frame(frame, confidence: float = None) -> str:
    """
    Save a frame that contains uncertain detections for later manual labeling.
    Returns the saved filename.
    """
    timestamp = int(time.time() * 1000)
    filename = f"uncertain_{timestamp}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)

    cv2.imwrite(filepath, frame)

    # Log the entry
    entry = {
        "file": filename,
        "timestamp": timestamp,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "labeled": False,
    }

    log = _load_log()
    log.append(entry)
    _save_log(log)

    msg = f"[Uncertain] Saved: {filename}"
    if confidence is not None:
        msg += f"  (conf={confidence:.3f})"
    print(msg)

    return filepath


def get_unlabeled_frames() -> list:
    """Return list of frame filenames that have not yet been labeled."""
    log = _load_log()
    return [e["file"] for e in log if not e.get("labeled", False)]


def mark_as_labeled(filename: str):
    """Mark a frame as labeled in the log."""
    log = _load_log()
    for entry in log:
        if entry["file"] == filename:
            entry["labeled"] = True
    _save_log(log)