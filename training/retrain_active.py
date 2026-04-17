from ultralytics import YOLO
import os
import shutil
import glob


# ─── Config ───────────────────────────────────────────
YAML_PATH      = os.path.abspath("data/yaml/bridge.yaml")
UNCERTAIN_DIR  = "data/uncertain"
OUTPUT_DIR     = "models/trained_model"
EPOCHS         = 20
IMG_SIZE       = 640
BATCH          = 8
# ──────────────────────────────────────────────────────


def find_best_model():
    """
    Look for best.pt in order of preference:
    1. models/trained_model/weights/best.pt  (our stable copy)
    2. Most recent run in runs/train/
    3. Base yolov8n.pt as last resort
    """
    candidates = [
        os.path.join(OUTPUT_DIR, "weights", "best.pt"),
        *sorted(
            glob.glob("runs/train/*/weights/best.pt"),
            key=os.path.getmtime,
            reverse=True,
        ),
        "yolov8n.pt",
    ]

    for path in candidates:
        if os.path.exists(path):
            print(f"[Retrain] Using model: {path}")
            return path

    raise FileNotFoundError(
        "No trained model found. Run option 3 (Train) first."
    )


def count_uncertain_samples():
    if not os.path.exists(UNCERTAIN_DIR):
        return 0
    return sum(1 for f in os.listdir(UNCERTAIN_DIR) if f.endswith(".jpg"))


def retrain():
    n_uncertain = count_uncertain_samples()
    if n_uncertain == 0:
        print("⚠️  No uncertain frames found in data/uncertain/")
        print("    Run live detection first (option 4) to collect uncertain samples.")
        print("    Then label them (option 6) before retraining.")
        return

    print(f"[Retrain] Found {n_uncertain} uncertain samples in {UNCERTAIN_DIR}")

    # Check if uncertain samples have labels
    label_dir = os.path.join(UNCERTAIN_DIR, "labels")
    if not os.path.exists(label_dir) or len(os.listdir(label_dir)) == 0:
        print("⚠️  Uncertain frames are not yet labeled.")
        print("    Run option 6 to label them first.")
        return

    model_path = find_best_model()
    model = YOLO(model_path)

    print(f"[Retrain] Epochs={EPOCHS}, ImgSize={IMG_SIZE}, Batch={BATCH}")

    model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        name="retrained_active",
        project="models",
        exist_ok=True,
    )

    # Copy new best weights over the stable path
    src = os.path.join("models", "retrained_active", "weights", "best.pt")
    dst = os.path.join(OUTPUT_DIR, "weights", "best.pt")

    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        print(f"\n✅ Retrain complete. Updated best model → {dst}")
    else:
        print("⚠️  Could not find retrained weights.")


if __name__ == "__main__":
    retrain()