import os
from ultralytics import YOLO

model_path = os.path.join("models", "violence_yolo.pt")
violence_model = YOLO(model_path)
print("Violence model loaded successfully from", model_path)