from ultralytics import YOLO
import os
import shutil

# ─── Config ───────────────────────────────────────────
MODEL_BASE   = "yolov8s.pt"
YAML_PATH    = os.path.abspath("data/yaml/bridge.yaml")
OUTPUT_DIR   = "models/trained_model"   # final resting place
EPOCHS       = 100
IMG_SIZE     = 640
BATCH        = 8
# ──────────────────────────────────────────────────────

def train():
    if not os.path.exists(YAML_PATH):
        raise FileNotFoundError(f"Dataset YAML not found: {YAML_PATH}")

    if not os.path.exists(MODEL_BASE):
        raise FileNotFoundError(
            f"Base model '{MODEL_BASE}' not found in project root. "
            "Download it with: python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\""
        )

    print(f"[Train] Using dataset: {YAML_PATH}")
    print(f"[Train] Epochs={EPOCHS}, ImgSize={IMG_SIZE}, Batch={BATCH}")

    model = YOLO(MODEL_BASE)

    results = model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        name="bridge_defect",
        project="runs/train",
        exist_ok=True,
    )

    # ── Copy best weights to a stable, well-known path ──
    src_weights = os.path.join("runs", "train", "bridge_defect", "weights")
    os.makedirs(os.path.join(OUTPUT_DIR, "weights"), exist_ok=True)

    for fname in ["best.pt", "last.pt"]:
        src = os.path.join(src_weights, fname)
        dst = os.path.join(OUTPUT_DIR, "weights", fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"[Train] Saved {fname} → {dst}")

    print(f"\n✅ Training complete. Best model at: {OUTPUT_DIR}/weights/best.pt")


if __name__ == "__main__":
    train()
else:
    train()