"""
Manual labeling tool for uncertain frames collected during live detection.
Shows each unlabeled frame and lets you draw bounding boxes.

Controls:
  Left-click + drag  → Draw bounding box
  0–9                → Assign class ID to last drawn box
  s                  → Save labels for this frame and move to next
  d                  → Discard this frame (delete it)
  q                  → Quit labeling session
"""

import cv2
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from active_learning.uncertainty_sampler import get_unlabeled_frames, mark_as_labeled

UNCERTAIN_DIR = "data/uncertain"
LABEL_DIR     = os.path.join(UNCERTAIN_DIR, "labels")
os.makedirs(LABEL_DIR, exist_ok=True)

# Load class names from bridge.yaml
def load_class_names():
    try:
        import yaml
        with open("data/yaml/bridge.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("names", {})
    except Exception as e:
        print(f"[Warning] Could not load class names from bridge.yaml ({e}). Using defaults.")
        return {0: "crack", 1: "spalling", 2: "corrosion", 3: "efflorescence"}


CLASS_NAMES = load_class_names()

# ── State for mouse drawing ──────────────────────────
drawing    = False
ix, iy     = -1, -1
boxes      = []         # list of [x1,y1,x2,y2]
current_box = None


def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, current_box, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        current_box = [x, y, x, y]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_box = [ix, iy, x, y]

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_box = [ix, iy, x, y]
        boxes.append({"box": list(current_box), "class_id": 0})
        print(f"  Box added (class=0 default). Press 0-9 to change class.")


def draw_boxes(frame, boxes, current_box=None):
    vis = frame.copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b["box"]
        cid = b["class_id"]
        label = CLASS_NAMES.get(cid, str(cid))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{cid}:{label}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if current_box:
        x1, y1, x2, y2 = current_box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 1)
    return vis


def to_yolo(box, img_w, img_h):
    x1, y1, x2, y2 = box
    x_center = max(0.0, min(1.0, (x1 + x2) / 2.0 / img_w))
    y_center = max(0.0, min(1.0, (y1 + y2) / 2.0 / img_h))
    w        = max(0.0, min(1.0, abs(x2 - x1) / img_w))
    h        = max(0.0, min(1.0, abs(y2 - y1) / img_h))
    return x_center, y_center, w, h


def label_frames():
    global boxes, current_box

    pending = get_unlabeled_frames()

    if not pending:
        print("✅ No unlabeled uncertain frames found.")
        return

    print(f"\n[Label] {len(pending)} frames to label.")
    print("Controls: drag=draw box | 0-9=set class | s=save | d=discard | q=quit\n")

    cv2.namedWindow("Label Uncertain Frames")
    cv2.setMouseCallback("Label Uncertain Frames", mouse_callback)

    try:
        for fname in pending:
            fpath = os.path.join(UNCERTAIN_DIR, fname)
            if not os.path.exists(fpath):
                mark_as_labeled(fname)
                continue

            frame = cv2.imread(fpath)
            if frame is None:
                mark_as_labeled(fname)
                continue

            img_h, img_w = frame.shape[:2]
            boxes = []
            current_box = None

            print(f"\n→ Frame: {fname}")

            while True:
                vis = draw_boxes(frame, boxes, current_box)

                # Instructions overlay
                instructions = [
                    f"File: {fname}",
                    f"Boxes: {len(boxes)}",
                    "drag=draw  0-9=class  s=save  d=discard  q=quit",
                ]
                for i, txt in enumerate(instructions):
                    cv2.putText(vis, txt, (10, 20 + i * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

                cv2.imshow("Label Uncertain Frames", vis)
                key = cv2.waitKey(20) & 0xFF

                # Change class of last drawn box
                if ord('0') <= key <= ord('9') and boxes:
                    cid = key - ord('0')
                    boxes[-1]["class_id"] = cid
                    print(f"  Last box → class {cid}: {CLASS_NAMES.get(cid, '?')}")

                elif key == ord('s'):
                    # Save YOLO label file
                    label_name = os.path.splitext(fname)[0] + ".txt"
                    label_path = os.path.join(LABEL_DIR, label_name)
                    with open(label_path, "w") as f:
                        for b in boxes:
                            cid = b["class_id"]
                            x, y, w, h = to_yolo(b["box"], img_w, img_h)
                            f.write(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                    mark_as_labeled(fname)
                    print(f"  ✅ Saved {len(boxes)} box(es) → {label_path}")
                    break

                elif key == ord('d'):
                    os.remove(fpath)
                    mark_as_labeled(fname)
                    print(f"  🗑  Discarded: {fname}")
                    break

                elif key == ord('q'):
                    print("\n[Label] Session ended.")
                    return

    finally:
        cv2.destroyAllWindows()

    print("\n✅ All frames labeled!")


if __name__ == "__main__":
    label_frames()