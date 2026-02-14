from ultralytics import YOLO
import os

model = YOLO("yolov8n.pt")

yaml_path = os.path.abspath("data/yaml/bridge.yaml")

model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=8,
    name="bridge_defect",
    exist_ok=True
)
