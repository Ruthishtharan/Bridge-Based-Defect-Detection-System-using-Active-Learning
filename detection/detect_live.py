from ultralytics import YOLO
import cv2
from camera.phone_camera import get_phone_camera
from active_learning.uncertainty_sampler import save_uncertain_frame, is_uncertain

MODEL_PATH = "models/trained_model/weights/best.pt"

model = YOLO(MODEL_PATH)

ip = input("Enter phone IP (example 192.168.1.5:8080): ")
cap = get_phone_camera(ip)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.25)

    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                conf = float(box.conf)

                if is_uncertain(conf):
                    save_uncertain_frame(frame, confidence=conf)

    annotated = results[0].plot()

    cv2.imshow("Bridge Defect Detection", annotated)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
