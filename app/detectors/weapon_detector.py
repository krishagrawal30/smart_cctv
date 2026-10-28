from collections import deque
from typing import Deque, List, Dict, Any

import numpy as np
import os

from config import CONFIG

try:
    from ultralytics import YOLO  # type: ignore
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None  # type: ignore
    _ULTRALYTICS_AVAILABLE = False


class WeaponDetector:
    """
    YOLOv26-based weapon detector for college campus cameras.

    Leverages YOLOv26's native NMS-free end-to-end inference and small-target
    awareness (STAL) to detect concealed and brandished weapons at edge camera latency.
    Includes multi-frame confirmation to prevent false positives during
    fights (e.g., fists/limbs misclassified as weapons).
    """

    # A weapon must be detected in at least this many of last N frames
    CONFIRM_WINDOW = 6         # Lowered for better responsiveness
    CONFIRM_MIN_HITS = 2       # Base hits required
 
    # Minimum bounding box area (pixels)
    MIN_BBOX_AREA = 800
    
    # Maximum bounding box fraction of total frame area (exclude person-sized boxes)
    MAX_BBOX_FRACTION = 0.20

    @staticmethod
    def _weapon_category(label: str) -> str:
        """Map class names to high-level weapon categories for consistent tracking."""
        lbl = label.lower()
        if any(k in lbl for k in ["knife", "blade", "dagger", "sharp", "machete", "sword"]):
            return "edged"
        if any(k in lbl for k in ["gun", "pistol", "rifle", "handgun", "firearm", "revolver"]):
            return "firearm"
        if "grenade" in lbl:
            return "explosive"
        return "weapon"

    def __init__(self) -> None:
        self.conf_threshold = CONFIG["detection"]["weapon_conf_threshold"]
        self.model_path = CONFIG["models"]["weapon_yolo_weights"]
        self.device = CONFIG["models"]["device"]
        self.model = None

        # Rolling history of raw detections for confirmation logic
        self._detection_history: Deque[List[Dict[str, Any]]] = deque(maxlen=self.CONFIRM_WINDOW)
        # Store last seen non-empty detections to bridge frames while confirmed
        self._last_valid_dets: List[Dict[str, Any]] = []

        if not _ULTRALYTICS_AVAILABLE:
            print("[WeaponDetector] ultralytics not installed; running in dummy mode.")
            return

        if not os.path.exists(self.model_path):
            print(f"[WeaponDetector] Weapon YOLO weights not found: {self.model_path} (SAFE mode)")
            self.model = None
            return

        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"[WeaponDetector] Loaded YOLOv26 weights from {self.model_path}")
        except Exception as e:  # noqa: BLE001
            print(f"[WeaponDetector] Could not load YOLO model: {e}")
            self.model = None

    def _raw_detect(self, frame: np.ndarray, fight_prob: float = 0.0) -> List[Dict[str, Any]]:
        """
        Run YOLO inference and return raw detections (before confirmation).
        """
        if self.model is None:
            return []

        # Use the standard model threshold
        current_conf = self.conf_threshold

        try:
            results = self.model.predict(
                source=frame,
                conf=current_conf,
                device=self.device,
                verbose=False,
                imgsz=640,    # Higher resolution for better small-object detection
            )
        except Exception as e:  # noqa: BLE001
            print(f"[WeaponDetector] Inference failed: {e}")
            return []

        detections: List[Dict[str, Any]] = []
        frame_h, frame_w = frame.shape[:2] if hasattr(frame, "shape") and len(frame.shape) >= 2 else (480, 640)
        max_bbox_area = frame_h * frame_w * self.MAX_BBOX_FRACTION

        for r in results:
            boxes = r.boxes
            names = r.names

            for box in boxes:
                cls_id = int(box.cls.item())
                label = names.get(cls_id, str(cls_id))
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox_area = (x2 - x1) * (y2 - y1)

                # Filter by weapon-like labels (matches all trained model classes including Grenade)
                weapon_keywords = ["gun", "weapon", "pistol", "rifle", "knife",
                                   "dagger", "sharp", "handgun", "blade",
                                   "machete", "sword", "firearm", "revolver", "grenade"]
                
                if not any(k in label.lower() for k in weapon_keywords):
                    continue

                # Filter by bounding box area (avoid person-sized "weapons" or microscopic noise)
                if bbox_area < self.MIN_BBOX_AREA:
                    continue
                
                if bbox_area > max_bbox_area:
                    continue

                # Calibrated dynamic thresholding:
                # When fight probability is high, increase required confidence moderately
                # to prevent flailing limbs from generating false weapon alerts.
                dynamic_conf = current_conf
                if fight_prob >= 0.50:
                    dynamic_conf = max(current_conf, 0.70)
                elif fight_prob >= 0.20:
                    dynamic_conf = max(current_conf, 0.60)

                if conf < dynamic_conf:
                    continue

                detections.append(
                    {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "label": label,
                        "confidence": conf,
                    }
                )

        return detections

    def detect_weapons(self, frame: np.ndarray, fight_prob: float = 0.0) -> List[Dict[str, Any]]:
        """
        Run YOLOv26 on a BGR frame with multi-frame confirmation.

        A weapon is only reported if it has been detected across sufficient
        recent frames within CONFIRM_WINDOW to filter single-frame artifacts.
        """
        raw_dets = self._raw_detect(frame, fight_prob=fight_prob)
        self._detection_history.append(raw_dets)

        def get_required_hits(label: str) -> int:
            cat = self._weapon_category(label)
            if fight_prob >= 0.50:
                return min(self.CONFIRM_WINDOW, 4)
            if cat == "edged":
                return 2
            return 3

        # Process each current detection against its category history requirement
        final_dets = []
        for d in raw_dets:
            label = d["label"]
            req = get_required_hits(label)
            cat = self._weapon_category(label)
            
            # Count how many of the recent frames had a weapon in the same category
            hits = sum(1 for frame_dets in self._detection_history 
                       if any(self._weapon_category(fd["label"]) == cat for fd in frame_dets))
            
            if hits >= req:
                final_dets.append(d)
                print(f"[WeaponDetector CONFIRMED] {label} ({d['confidence']:.2f}) with {hits}/{req} hits")

        if final_dets:
            self._last_valid_dets = final_dets
            return final_dets
        else:
            # PERSISTENCE: If we were confirmed in the very recent past, stay alert for a moment
            confirm_frames = sum(1 for frame_dets in self._detection_history if len(frame_dets) > 0)
            
            # Do NOT persist gun/rifle detections if a fight is happening, to let false flags drop instantly
            persistence_allowed = True
            if self._last_valid_dets and fight_prob > 0.15:
                if any(any(k in d["label"].lower() for k in ["gun", "rifle", "pistol", "firearm"]) for d in self._last_valid_dets):
                    persistence_allowed = False

            if persistence_allowed and confirm_frames >= 3 and self._last_valid_dets:
                print(f"[WeaponDetector PERSISTED] Showing last seen {len(self._last_valid_dets)} box(es) — still confirmed ({confirm_frames} total hits)")
                return self._last_valid_dets
            else:
                self._last_valid_dets = []
                if raw_dets:
                    print(f"[WeaponDetector REJECTED] {len(raw_dets)} detection(s) — failed hit requirements")
                return []
