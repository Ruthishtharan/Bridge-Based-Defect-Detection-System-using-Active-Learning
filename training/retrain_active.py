from ultralytics import YOLO

model = YOLO("models/trained_model/weights/best.pt")

model.train(
    data="data/yaml/bridge.yaml",
    epochs=20,
    imgsz=256,
    batch=16,
    name="retrained_model",
    project="models",
    exist_ok=True,
)

print("Retraining complete. Model saved to models/retrained_model/")
