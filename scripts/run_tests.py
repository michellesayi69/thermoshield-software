import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from detect_threat import detect_threat, BODY_TEMP_MIN, COLD_SPOT_MIN, COLD_SPOT_MAX


def make_frame(body_pixels=0, cold_pixels=0):
    """
    Create a deterministic thermal frame.
    - Background: exactly 22.0°C
    - Body pixels: exactly 33.0°C (well above 28°C threshold)
    - Cold pixels: exactly 19.0°C (well below 25°C threshold)
    - Body and cold pixel sets are guaranteed non-overlapping
    """
    frame = np.full(768, 22.0)

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

print("=" * 78)
print("THERMOSHIELD — TEST RESULTS")
print("=" * 78)

# Regression: an all-background frame (22 C) must be counted as 0 body
# and 0 cold pixels under the algorithm's actual thresholds. This
# catches the 25.0-vs-21.0 bug that made the diagnostic disagree with
# detect_threat().
assert sum(1 for t in make_frame(0, 0) if t >= BODY_TEMP_MIN) == 0
assert sum(1 for t in make_frame(0, 0) if COLD_SPOT_MIN <= t <= COLD_SPOT_MAX) == 0

passed = 0
for name, frame, expected in tests:
    body = sum(1 for t in frame if t >= BODY_TEMP_MIN)
    cold = sum(1 for t in frame if COLD_SPOT_MIN <= t <= COLD_SPOT_MAX)
    result = detect_threat(frame)
    status = "PASS" if result == expected else "FAIL"
    if status == "PASS":
        passed += 1
    print(f"{name:25} | body={body:3d} cold={cold:3d} | Expected: {expected:17} | Got: {result:17} | {status}")

print("=" * 78)
print(f"Passed: {passed} / {len(tests)}")
print("=" * 78)