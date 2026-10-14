import threading
import time
from typing import Generator, Tuple, Optional, Union

import cv2
import numpy as np

from config import CONFIG


class CampusCameraStream:
    """
    Handles video capture from a college campus camera (corridor, hostel gate, etc.)
    using a background thread to maintain a real-time (lowest latency) buffer.
    """

    def __init__(self, source: Optional[Union[int, str]] = None) -> None:
        cam_cfg = CONFIG["camera"]
        self.source: Union[int, str] = source if source is not None else cam_cfg["source"]
        self.width = cam_cfg["frame_width"]
        self.height = cam_cfg["frame_height"]
        
        # Reverting to simplest initialization for maximum stability
        # Force DirectShow on Windows to avoid MSMF error -1072873821
        is_device_idx = isinstance(self.source, int) or (isinstance(self.source, str) and str(self.source).isdigit())
        if is_device_idx:
            self.cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        else:
            self.cap = cv2.VideoCapture(self.source)
            
        if not self.cap.isOpened():
            print(f"[CampusCameraStream] WARNING: Unable to open campus camera source={self.source}")

        self.last_frame: Optional[np.ndarray] = None
        self.success: bool = False
        self.new_frame_ready: bool = False  # Track when a FRESH frame arrives
        self.running: bool = True
        self.lock = threading.Lock()
        
        # Start background thread to keep buffer empty
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self) -> None:
        """Continuously grab frames in a background thread."""
        while self.running:
            if self.cap is not None and self.cap.isOpened():
                success, frame = self.cap.read()
                if success:
                    with self.lock:
                        self.success = True
                        self.last_frame = frame
                        self.new_frame_ready = True
                else:
                    # Brief rest on capture failure
                    time.sleep(0.1)
            else:
                time.sleep(0.1)

    def frames(self) -> Generator[Tuple[bool, Optional[np.ndarray]], None, None]:
        """
        Yield (success, frame) ONLY when a new frame is ready.
        This prevents the pipeline from processing the same frame 100x slower.
        """
        while self.running:
            frame_to_yield = None
            success_to_yield = False
            
            with self.lock:
                if self.new_frame_ready:
                    frame_to_yield = self.last_frame
                    success_to_yield = self.success
                    self.new_frame_ready = False
            
            if frame_to_yield is not None:
                yield success_to_yield, frame_to_yield
            else:
                # Rest the loop for a tiny bit (1ms) to avoid CPU spin while waiting
                time.sleep(0.001)

    def release(self) -> None:
        """Release the campus camera resource."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        with self.lock:
            self.last_frame = None
            self.new_frame_ready = False


