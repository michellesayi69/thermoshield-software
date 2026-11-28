"""Tests for background-aware detection (Stage 4, sub-step 3).

The defining test: at 18 C ambient, the baseline classifier misclassifies
a person as THERMAL ANOMALY because the background falls inside the cold
band. Background-aware detection correctly returns PERSON.

Proof that the design addresses the README's explicit Limitations concern.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from thermoshield import (
    classify_frame,
    classify_frame_with_confidence,
    DetectionConfig,
    estimate_background,
    FRAME_SIZE,
)


def make_scene(ambient, body_pixels=0, cold_pixels=0):
    """Deterministic frame at given ambient with body/cold pixel counts."""
    frame = np.full(FRAME_SIZE, float(ambient))
    for i in range(body_pixels):
        frame[i] = float(ambient) + 11.0  # 11 C above ambient
    for i in range(body_pixels, body_pixels + cold_pixels):
        frame[i] = float(ambient) - 5.0   # 5 C below ambient
    return frame


BG = DetectionConfig(background_aware=True)
results = []

# 1. estimate_background returns the median.
f = make_scene(22.0, body_pixels=100)
bg = estimate_background(f)
results.append((
    "1. estimate_background == 22",
    "PASS" if abs(bg - 22.0) < 0.01 else f"FAIL (got {bg})",
))

# 2. Default (background_aware=False) at 22 C matches baseline exactly.
f = make_scene(22.0, body_pixels=100, cold_pixels=20)
lab_baseline = classify_frame(f)
lab_bg = classify_frame(f, config=BG)
results.append((
    "2. 22 C: baseline and bg-aware agree",
    "PASS" if lab_baseline == lab_bg == "THERMAL ANOMALY"
    else f"FAIL (baseline={lab_baseline}, bg={lab_bg})",
))

# 3. Empty room at 18 C: baseline sees every pixel as cold; bg-aware sees none.
f = make_scene(18.0, body_pixels=0, cold_pixels=0)
lab_baseline = classify_frame(f)
lab_bg = classify_frame(f, config=BG)
results.append((
    "3. 18 C empty room: both CLEAR",
    "PASS" if lab_baseline == lab_bg == "CLEAR"
    else f"FAIL (baseline={lab_baseline}, bg={lab_bg})",
))

# 4. Person at 18 C: baseline MISFIRES (background in cold band);
#    bg-aware correctly returns PERSON.
f = make_scene(18.0, body_pixels=100, cold_pixels=0)
lab_baseline = classify_frame(f)
lab_bg = classify_frame(f, config=BG)
results.append((
    "4. 18 C person: baseline misfires, bg correct",
    "PASS" if lab_baseline == "THERMAL ANOMALY" and lab_bg == "PERSON"
    else f"FAIL (baseline={lab_baseline}, bg={lab_bg})",
))

# 5. Person + real cold region at 18 C: both should say THERMAL ANOMALY.
f = make_scene(18.0, body_pixels=100, cold_pixels=20)
lab_baseline = classify_frame(f)
lab_bg = classify_frame(f, config=BG)
results.append((
    "5. 18 C person+cold: both ANOMALY",
    "PASS" if lab_baseline == lab_bg == "THERMAL ANOMALY"
    else f"FAIL (baseline={lab_baseline}, bg={lab_bg})",
))

# 6. Person at 26 C ambient: baseline may miss body pixels (28 C not so far);
#    bg-aware detects them as delta>=6.
f = make_scene(26.0, body_pixels=100, cold_pixels=0)
lab_baseline = classify_frame(f)
lab_bg = classify_frame(f, config=BG)
results.append((
    "6. 26 C person: bg aware detects, baseline may miss",
    "PASS" if lab_bg == "PERSON" else f"FAIL (bg={lab_bg})",
))

# 7. Confidence also respects background_aware.
f = make_scene(18.0, body_pixels=100, cold_pixels=0)
label, conf = classify_frame_with_confidence(f, config=BG)
results.append((
    "7. confidence respects bg-aware",
    "PASS" if label == "PERSON" and 0 <= conf <= 100
    else f"FAIL (label={label}, conf={conf})",
))

# 8. Custom delta thresholds work independently of absolute thresholds.
#    Very strict body delta should reject person-like frame at 22 C.
strict = DetectionConfig(background_aware=True, body_delta_min=20.0)
f = make_scene(22.0, body_pixels=100, cold_pixels=0)
lab = classify_frame(f, config=strict)
results.append((
    "8. custom delta rejected by strict threshold",
    "PASS" if lab == "CLEAR" else f"FAIL (got {lab})",
))

print("=" * 70)
print("THERMOSHIELD - BACKGROUND-AWARE TESTS (Stage 4, sub-step 3)")
print("=" * 70)
passed = 0
for name, status in results:
    print(f"{name:48} | {status}")
    if status == "PASS":
        passed += 1
print("=" * 70)
print(f"Passed: {passed} / {len(results)}")
print("=" * 70)
sys.exit(0 if passed == len(results) else 1)
