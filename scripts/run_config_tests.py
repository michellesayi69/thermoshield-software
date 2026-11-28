"""Tests for the configurable DetectionConfig (Stage 4).

Complements run_tests.py. Those eight tests prove the default config
reproduces the baseline fixed-threshold behaviour. These seven prove
that custom configurations actually alter the classifier, that the
confidence function respects config, and that invalid configs are
rejected at construction time.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from thermoshield import (
    classify_frame,
    classify_frame_with_confidence,
    DetectionConfig,
    DEFAULT_CONFIG,
    FRAME_SIZE,
)


def make_frame(body_pixels=0, cold_pixels=0):
    """22 C background, 33 C body pixels, 19 C cold pixels."""
    frame = np.full(FRAME_SIZE, 22.0)
    for i in range(body_pixels):
        frame[i] = 33.0
    for i in range(body_pixels, body_pixels + cold_pixels):
        frame[i] = 19.0
    return frame


def expect_raises(fn, exc_type, label):
    try:
        fn()
    except exc_type:
        return (label, "PASS")
    except Exception as e:
        return (label, f"FAIL (wrong exception: {type(e).__name__})")
    return (label, "FAIL (did not raise)")


results = []

# 1. Default config == no config: baseline preserved.
f = make_frame(100, 20)
implicit = classify_frame(f)
explicit = classify_frame(f, config=DEFAULT_CONFIG)
results.append((
    "1. default config == implicit",
    "PASS" if implicit == explicit else f"FAIL ({implicit} vs {explicit})",
))

# 2. Custom min_body_pixels changes the label.
f = make_frame(30, 0)  # 30 body, 0 cold: CLEAR at default, PERSON at min=20
default = classify_frame(f)
custom = classify_frame(f, config=DetectionConfig(min_body_pixels=20))
results.append((
    "2. custom min_body_pixels",
    "PASS" if default == "CLEAR" and custom == "PERSON"
    else f"FAIL (default={default}, custom={custom})",
))

# 3. Custom cold band picks up 22 C pixels as cold.
f = make_frame(100, 0)  # 100 body, 668 at 22 C (not in default cold band)
default = classify_frame(f)
wider = DetectionConfig(cold_spot_max=25.0, min_cold_pixels=100)
custom = classify_frame(f, config=wider)
results.append((
    "3. custom cold band",
    "PASS" if default == "PERSON" and custom == "THERMAL ANOMALY"
    else f"FAIL (default={default}, custom={custom})",
))

# 4-6. Invalid configs are rejected at construction.
results.append(expect_raises(
    lambda: DetectionConfig(cold_spot_min=25.0, cold_spot_max=20.0),
    ValueError, "4. invalid: cold_min >= cold_max"))
results.append(expect_raises(
    lambda: DetectionConfig(body_temp_min=20.0),
    ValueError, "5. invalid: body <= cold_max"))
results.append(expect_raises(
    lambda: DetectionConfig(min_body_pixels=-1),
    ValueError, "6. invalid: negative pixel count"))

# 7. Confidence function also respects config.
f = make_frame(30, 0)
label, conf = classify_frame_with_confidence(
    f, config=DetectionConfig(min_body_pixels=20)
)
results.append((
    "7. confidence respects config",
    "PASS" if label == "PERSON" else f"FAIL (label={label})",
))

print("=" * 70)
print("THERMOSHIELD - CONFIG TESTS (Stage 4)")
print("=" * 70)
passed = 0
for name, status in results:
    print(f"{name:32} | {status}")
    if status == "PASS":
        passed += 1
print("=" * 70)
print(f"Passed: {passed} / {len(results)}")
print("=" * 70)
sys.exit(0 if passed == len(results) else 1)
