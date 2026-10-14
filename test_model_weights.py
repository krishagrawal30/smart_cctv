import os
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def test_models():
    if YOLO is None:
        print("SKIP: ultralytics not installed. Install ultralytics to run this test.")
        return

    models = [
        ("Violence YOLO", "models/violence_yolo.pt"),
        ("Weapon YOLO", "models/weapon_yolo.pt"),
    ]

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for name, path in models:
        print(f"\n--- Testing {name} ({path}) ---")
        if not os.path.exists(path):
            print(f"SKIP: Model file not found: {path}")
            continue

        model = YOLO(path)
        print(f"Task: {model.task}")
        print(f"Classes: {model.names}")

        # Run test inference on dummy frame
        results = model.predict(source=dummy_frame, verbose=False)
        print(f"Inference output: {len(results)} result container(s) returned [PASS]")


if __name__ == "__main__":
    test_models()
