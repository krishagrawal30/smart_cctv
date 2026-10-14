import cv2
import urllib.request
import numpy as np
import time
import urllib.error

# Detectors may require heavy dependencies (ultralytics/torch). Import lazily.
# We'll attempt to import them inside `run_tests()` and skip tests if unavailable.

def run_tests():
    # Lazy imports to avoid hard dependency requirements during quick checks
    try:
        from app.detectors.weapon_detector import WeaponDetector
    except Exception:
        WeaponDetector = None

    try:
        from app.detectors.fight_detector import FightDetector
    except Exception:
        FightDetector = None

    try:
        from app.risk.risk_engine import RiskEngine, RiskLevel
    except Exception:
        RiskEngine = None
        RiskLevel = None

    # 1. Test Weapon Detector with synthetic test frame
    print("\n--- [1] Testing Weapon Detector ---")
    if WeaponDetector is None:
        print("SKIP: WeaponDetector unavailable (missing dependencies or model).")
    else:
        w_detector = WeaponDetector()
        # Create a synthetic 640x480 test image to verify inference pipeline
        synthetic_img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add simple geometric contrast patterns
        cv2.rectangle(synthetic_img, (200, 150), (440, 330), (200, 200, 200), -1)
        dets = w_detector.detect_weapons(synthetic_img)
        print(f"Synthetic frame inference completed: detected {len(dets)} weapons (pipeline functional).")

    # 2. Test Fight Detector with synthetic sequence
    print("\n--- [2] Testing Fight Detector ---")
    if FightDetector is None:
        print("SKIP: FightDetector unavailable (missing dependencies or model).")
    else:
        f_detector = FightDetector()
        print(f"Smoothing window: {f_detector.SMOOTHING_WINDOW}")
        np.random.seed(42)
        # Run predictions to test smoothing accumulation
        for i in range(5):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            p = f_detector.predict_fight(frame)
            print(f"  Prediction {i+1}: raw={f_detector.last_raw_prob:.4f} smooth={f_detector.last_smoothed_prob:.4f}")
        
        # Test reset
        f_detector.reset()
        assert f_detector.last_smoothed_prob == 0.0, "Reset should clear smoothed prob"
        print(f"After reset: prob={f_detector.last_smoothed_prob:.4f} [PASS]")

    # 3. Test Risk Engine Logic
    print("\n--- [3] Testing Risk Engine Logic ---")
    if RiskEngine is None:
        print("ERROR: RiskEngine unavailable — cannot run risk logic tests.")
        return
    engine = RiskEngine()
    
    # Test no risk
    res1 = engine.update_and_evaluate(fight_prob=0.1, weapon_detections=[], current_time=0.0)
    assert res1.level == RiskLevel.NO_RISK, f"Expected NO_RISK, got {res1.level}"
    print(f"Test 1 (No incidents): {res1.level.value} [PASS]")
    
    # Test medium risk (fight only)
    res2 = engine.update_and_evaluate(fight_prob=0.85, weapon_detections=[], current_time=1.0)
    assert res2.level == RiskLevel.MEDIUM_RISK, f"Expected MEDIUM_RISK, got {res2.level}"
    print(f"Test 2 (Fight only): {res2.level.value} [PASS]")
    
    # Test high risk (weapon only)
    engine.reset()
    res3 = engine.update_and_evaluate(fight_prob=0.1, weapon_detections=[{"label": "Gun", "confidence": 0.9}], current_time=2.0)
    assert res3.level == RiskLevel.HIGH_RISK, f"Expected HIGH_RISK, got {res3.level}"
    print(f"Test 3 (Weapon only): {res3.level.value} [PASS]")
    
    # Test critical risk (weapon + fight)
    engine.reset()
    res4 = engine.update_and_evaluate(fight_prob=0.9, weapon_detections=[{"label": "Gun", "confidence": 0.9}], current_time=3.0)
    assert res4.level == RiskLevel.CRITICAL, f"Expected CRITICAL, got {res4.level}"
    print(f"Test 4 (Weapon + Fight): {res4.level.value} [PASS]")
    
    # Test prolonged fight escalation deterministically
    print("\nTesting prolonged fight (escalates to CRITICAL over time):")
    engine.reset()
    t_start = 100.0
    # At t=0s, fight starts -> MEDIUM_RISK
    r_start = engine.update_and_evaluate(fight_prob=0.85, weapon_detections=[], current_time=t_start)
    print(f"  t=0.0s: {r_start.level.value}")
    assert r_start.level == RiskLevel.MEDIUM_RISK, "Fight at t=0s should be MEDIUM_RISK"
    
    # At t=2.0s (< 4.0s min_duration) -> MEDIUM_RISK
    r_mid = engine.update_and_evaluate(fight_prob=0.85, weapon_detections=[], current_time=t_start + 2.0)
    print(f"  t=2.0s: {r_mid.level.value}")
    assert r_mid.level == RiskLevel.MEDIUM_RISK, "Fight at t=2.0s should still be MEDIUM_RISK"
    
    # At t=4.5s (>= 4.0s min_duration) -> CRITICAL
    r_crit = engine.update_and_evaluate(fight_prob=0.85, weapon_detections=[], current_time=t_start + 4.5)
    print(f"  t=4.5s: {r_crit.level.value}")
    assert r_crit.level == RiskLevel.CRITICAL, "Prolonged fight at t=4.5s should escalate to CRITICAL"
    print("Prolonged fight escalation test: [PASS]")

if __name__ == "__main__":
    run_tests()
