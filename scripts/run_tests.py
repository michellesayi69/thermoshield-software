"""Run the eight deterministic ThermoShield detection tests.

Test cases, expected labels, and frame construction are unchanged from
the original script. Only the imports have moved to the thermoshield
package and the diagnostic counts against the algorithm's thresholds.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from thermoshield import (
    classify_frame,
    BODY_TEMP_MIN,
    COLD_SPOT_MIN,
    COLD_SPOT_MAX,
    FRAME_SIZE,
)


def make_frame(body_pixels=0, cold_pixels=0):
    """Create a deterministic thermal frame.

    - Background: exactly 22.0 C (above the 21 C cold-band ceiling)
    - Body pixels: exactly 33.0 C (well above the 28 C body threshold)
    - Cold pixels: exactly 19.0 C (inside the 15-21 C cold band)
    - Body and cold pixel sets are guaranteed non-overlapping
    """
    frame = np.full(FRAME_SIZE, 22.0)
    for i in range(body_pixels):
        frame[i] = 33.0
    for i in range(body_pixels, body_pixels + cold_pixels):
        frame[i] = 19.0
    return frame


tests = [
    ("1. Empty room",          make_frame(0, 0),     "CLEAR"),
    ("2. Person standing",     make_frame(100, 0),   "PERSON"),
    ("3. Cold spoon on chest", make_frame(100, 20),  "THERMAL ANOMALY"),
    ("4. Phone in pocket",     make_frame(100, 0),   "PERSON"),
    ("5. Cold bottle at side", make_frame(100, 20),  "THERMAL ANOMALY"),
    ("6. Person too far",      make_frame(30, 0),    "CLEAR"),
    ("7. Two people",          make_frame(200, 0),   "PERSON"),
    ("8. Backpack",            make_frame(100, 10),  "THERMAL ANOMALY"),
]

# Regression: empty frame must report 0 body and 0 cold pixels.
assert sum(1 for t in make_frame(0, 0) if t >= BODY_TEMP_MIN) == 0
assert sum(1 for t in make_frame(0, 0) if COLD_SPOT_MIN <= t <= COLD_SPOT_MAX) == 0

print("=" * 78)
print("THERMOSHIELD - TEST RESULTS")
print("=" * 78)

passed = 0
for name, frame, expected in tests:
    body = sum(1 for t in frame if t >= BODY_TEMP_MIN)
    cold = sum(1 for t in frame if COLD_SPOT_MIN <= t <= COLD_SPOT_MAX)
    result = classify_frame(frame)
    status = "PASS" if result == expected else "FAIL"
    if status == "PASS":
        passed += 1
    print(f"{name:25} | body={body:3d} cold={cold:3d} | Expected: {expected:17} | Got: {result:17} | {status}")

print("=" * 78)
print(f"Passed: {passed} / {len(tests)}")
print("=" * 78)
