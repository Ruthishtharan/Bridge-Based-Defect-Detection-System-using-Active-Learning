from ultralytics import YOLO
import cv2
import os
import glob
from camera.phone_camera import get_phone_camera
from active_learning.uncertainty_sampler import save_uncertain_frame, is_uncertain

# ─── Config ───────────────────────────────────────────
CONF_THRESHOLD = 0.25
WINDOW_TITLE   = "Bridge Defect Detection  [ESC = quit]"
# ──────────────────────────────────────────────────────


def find_model():
    """Find best available trained model."""
    candidates = [
        os.path.join("models", "trained_model", "weights", "best.pt"),
        *sorted(
            glob.glob("runs/train/*/weights/best.pt"),
            key=os.path.getmtime,
            reverse=True,
        ),
        # Also check the old runs/detect path (existing training results)
        os.path.join("runs", "detect", "bridge_defect", "weights", "best.pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"[Detect] Loaded model: {p}")
            return p
    raise FileNotFoundError(
        "No trained model found. Run option 3 (Train) first."
    )


def get_camera():
    """Ask user for camera source: phone IP or local webcam."""
    print("\nCamera Source:")
    print("  1 → Phone IP Camera (DroidCam / IP Webcam app)")
    print("  2 → Local Webcam (laptop / USB camera)")
    choice = input("Select [1/2]: ").strip()

    if choice == "1":
        ip = input("Enter phone IP:PORT (e.g. 192.168.1.5:8080): ").strip()
        cap = get_phone_camera(ip)
        if not cap.isOpened():
            print(f"⚠️  Cannot connect to {ip}. Falling back to local webcam.")
            cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("❌ No camera source available.")

    return cap


def run_detection():
    model_path = find_model()
    model = YOLO(model_path)

    cap = get_camera()

    uncertain_count = 0
    frame_count = 0

    print("\n[Detect] Running... Press ESC to stop.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Lost camera feed.")
                break

            frame_count += 1
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)

            # ── Active learning: save uncertain detections ──
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        conf = float(box.conf)
                        if is_uncertain(conf):
                            save_uncertain_frame(frame, confidence=conf)
                            uncertain_count += 1

            # ── Annotate and display ──
            if results and results[0] is not None:
                annotated = results[0].plot()
            else:
                annotated = frame

            # Overlay stats
            info = f"Frames: {frame_count}  |  Uncertain saved: {uncertain_count}"
            cv2.putText(
                annotated, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

            cv2.imshow(WINDOW_TITLE, annotated)

            if cv2.waitKey(1) & 0xFF == 27:   # ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n✅ Detection stopped. {uncertain_count} uncertain frames saved to data/uncertain/")


if __name__ == "__main__":
    run_detection()