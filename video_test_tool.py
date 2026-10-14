"""
Video Test Tool for Violence Detection
=======================================
Run the FightDetector on a video file to verify its performance
without needing the full live pipeline.

Usage:
    python video_test_tool.py path/to/video.mp4
    python video_test_tool.py 0                    # webcam
"""
import sys
import cv2
import numpy as np
import time

from app.detectors.fight_detector import FightDetector
from app.detectors.weapon_detector import WeaponDetector
from app.risk.risk_engine import RiskEngine
from app.utils.drawing import draw_weapon_boxes, draw_risk_overlay


def run_video_test(source):
    """
    Run fight and weapon detection on a video source and display annotated output.
    """
    # If source is a digit string, treat as camera index
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        return

    fight_detector = FightDetector()
    weapon_detector = WeaponDetector()
    risk_engine = RiskEngine()
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    if fps <= 0 or np.isnan(fps):
        fps = 20
    frame_delay = max(1, int(1000 / fps))

    print(f"[VideoTest] Source: {source}")
    print(f"[VideoTest] Fight threshold: {fight_detector.prob_threshold}")
    print(f"[VideoTest] Weapon threshold: {weapon_detector.conf_threshold}")
    print(f"[VideoTest] Smoothing window: {fight_detector.SMOOTHING_WINDOW}")
    print("[VideoTest] Press 'q' to quit, 'r' to reset buffers\n")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video for continuous testing
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            fight_detector.reset()
            risk_engine.reset()
            frame_idx = 0
            print("[VideoTest] --- Video looped, buffers reset ---")
            continue

        frame_idx += 1
        video_time = frame_idx / fps
        fight_prob = fight_detector.predict_fight(frame)
        weapon_dets = weapon_detector.detect_weapons(frame, fight_prob=fight_prob)

        risk = risk_engine.update_and_evaluate(
            fight_prob=fight_prob,
            weapon_detections=weapon_dets,
            current_time=video_time,
        )

        annotated = frame.copy()
        draw_weapon_boxes(annotated, weapon_dets)
        draw_risk_overlay(annotated, risk, "VIDEO TEST")

        # Extra debug info on frame
        debug_text = (
            f"Frame {frame_idx} | "
            f"Raw: {detector.last_raw_prob:.3f} | "
            f"Smooth: {detector.last_smoothed_prob:.3f}"
        )
        cv2.putText(
            annotated, debug_text, (10, annotated.shape[0] - 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
        )

        cv2.imshow("Violence Detection Test", annotated)

        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            detector.reset()
            print("[VideoTest] Buffers manually reset.")

        # Print periodic status
        if frame_idx % 30 == 0:
            print(
                f"  Frame {frame_idx:5d} | "
                f"Raw={detector.last_raw_prob:.3f} "
                f"Smooth={detector.last_smoothed_prob:.3f} "
                f"Risk={risk.level.value}"
            )

    cap.release()
    cv2.destroyAllWindows()
    print("\n[VideoTest] Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python video_test_tool.py <video_path_or_camera_index>")
        print("  e.g. python video_test_tool.py fight_clip.mp4")
        print("  e.g. python video_test_tool.py 0   (webcam)")
        sys.exit(1)

    run_video_test(sys.argv[1])
