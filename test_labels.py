import os

try:
	from ultralytics import YOLO
except Exception:
	YOLO = None

model_path = "models/violence_yolo.pt"

if YOLO is None:
	print("SKIP: ultralytics not installed; cannot list YOLO classes.")
else:
	if not os.path.exists(model_path):
		print(f"SKIP: YOLO model not found: {model_path}")
	else:
		model = YOLO(model_path)
		print("Classes:", model.names)
