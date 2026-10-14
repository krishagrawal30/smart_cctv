from collections import deque
from typing import Deque, Optional
import os

import cv2
import numpy as np

from config import CONFIG

try:
    from ultralytics import YOLO  # type: ignore
    _ULTRALYTICS_AVAILABLE = True
except Exception:
    YOLO = None  # type: ignore
    _ULTRALYTICS_AVAILABLE = False


class FightDetector:
    """
    YOLOv8 Violence Detection with confidence smoothing.

    Predicts the probability of a fight based on bounding box confidences,
    and smooths predictions over a short rolling window to prevent flickering.
    """

    # Smoothing window: reduced to 4 for faster reaction to sudden fights
    SMOOTHING_WINDOW = 4
    # Minimum inter-frame motion ratio to corroborate dynamic altercation activity
    MIN_MOTION_RATIO = 0.015

    def __init__(self) -> None:
        self.prob_threshold = CONFIG["detection"]["fight_prob_threshold"]
        self.weights_path = CONFIG["models"]["violence_model_weights"]

        self.model = None
        # Only attempt to load model if ultralytics is available
        if _ULTRALYTICS_AVAILABLE:
            self._load_model()
        else:
            print("[FightDetector] ultralytics not installed; running in dummy mode.")

        # Rolling window of recent probabilities for smoothing
        self._prob_history: Deque[float] = deque(maxlen=self.SMOOTHING_WINDOW)

        # Previous frame in grayscale for temporal motion gating
        self._prev_gray: Optional[np.ndarray] = None

        # Cache the last probabilities for debugging/overlay
        self.last_raw_prob: float = 0.0
        self.last_smoothed_prob: float = 0.0

    def _load_model(self) -> None:
        """
        Load Ultralytics YOLO violence detection model.
        """
        if not os.path.exists(self.weights_path):
            print(f"[FightDetector] Model not found: {self.weights_path} (SAFE mode)")
            self.model = None
            return

        try:
            self.model = YOLO(self.weights_path)
            print(f"[FightDetector] Loaded YOLO violence model: {self.weights_path}")
            print(f"[FightDetector] Classes: {self.model.names}")
        except Exception as e:
            print(f"[FightDetector] Failed to load model: {e}")
            self.model = None

    def predict_fight(self, frame: np.ndarray) -> float:
        """
        Runs YOLO inference on a single frame with temporal motion gating
        and returns smoothed fight probability.
        """
        if self.model is None or frame is None:
            return 0.0

        # Temporal Motion Gating:
        # Physical violence is an intrinsically dynamic event.
        # Compute inter-frame pixel difference to prevent static scenes from triggering alerts.
        has_motion = True
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(gray, (160, 120))
            if self._prev_gray is not None:
                diff = cv2.absdiff(small_gray, self._prev_gray)
                _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion_ratio = float(np.count_nonzero(motion_mask) / motion_mask.size)
                if motion_ratio < self.MIN_MOTION_RATIO:
                    has_motion = False
            self._prev_gray = small_gray
        except Exception:
            has_motion = True

        if not has_motion:
            raw_prob = 0.0
            self.last_raw_prob = 0.0
            self._prob_history.append(0.0)
            weights = list(range(1, len(self._prob_history) + 1))
            self.last_smoothed_prob = sum(w * p for w, p in zip(weights, self._prob_history)) / sum(weights)
            return self.last_smoothed_prob

        raw_prob = 0.0
        try:
            # inference natively on YOLO
            results = self.model.predict(
                source=frame,
                conf=0.10, # Very low threshold to capture raw probability baseline
                iou=0.45,
                verbose=False,
                device=CONFIG["models"]["device"]
            )
            
            # Find the highest confidence among any "VOILANCE" or similarly named class prediction
            max_fight_conf = 0.0
            max_non_fight_conf = 0.0
            
            for r in results:
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls_idx = int(box.cls[0].item())
                        label = self.model.names[cls_idx].upper()
                        conf = box.conf[0].item()
                        
                        # Support various spellings: VIOLENCE, VOILANCE (common typo in datasets), FIGHT, COMBAT
                        violence_keywords = ['VIOLENCE', 'VOILANCE', 'VOILLANCE', 'FIGHT', 'COMBAT']
                        is_violence = any(k in label for k in violence_keywords) and 'NON' not in label and 'NO_' not in label
                        
                        if is_violence:
                            if conf > max_fight_conf:
                                max_fight_conf = conf
                        else:
                            if conf > max_non_fight_conf:
                                max_non_fight_conf = conf

            # Hard suppression logic: If the model is more confident that the scene is NON_VOILANCE, 
            # we aggressively zero out the fight prediction to fix bad model training artifacts.
            if max_non_fight_conf > max_fight_conf:
                raw_prob = 0.0
            else:
                raw_prob = float(max_fight_conf)
        except Exception as e:
            print(f"[FightDetector ERROR] Inference failed: {e}")
            raw_prob = 0.0

        self.last_raw_prob = raw_prob

        # Push to smoothing window
        self._prob_history.append(raw_prob)

        # Weighted average — more recent frames count more
        weights = list(range(1, len(self._prob_history) + 1))
        total_w = sum(weights)
        smoothed = sum(w * p for w, p in zip(weights, self._prob_history)) / total_w

        self.last_smoothed_prob = smoothed

        return smoothed

    def reset(self) -> None:
        """Clear history — useful when switching cameras."""
        self._prob_history.clear()
        self._prev_gray = None
        self.last_raw_prob = 0.0
        self.last_smoothed_prob = 0.0