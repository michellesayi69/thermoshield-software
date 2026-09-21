import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from detect_threat import detect_threat


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
    ("3. Cold spoon on chest", make_frame(100, 20),  "WEAPON DETECTED"),
    ("4. Phone in pocket",     make_frame(100, 0),   "PERSON"),
    ("5. Cold bottle at side", make_frame(100, 20),  "WEAPON DETECTED"),
    ("6. Person too far",      make_frame(30, 0),    "CLEAR"),
    ("7. Two people",          make_frame(200, 0),   "PERSON"),
    ("8. Backpack",            make_frame(100, 10),  "WEAPON DETECTED"),
]

print("=" * 78)
print("THERMOSHIELD — TEST RESULTS")
print("=" * 78)

passed = 0
for name, frame, expected in tests:
    body = sum(1 for t in frame if t >= 28.0)
    cold = sum(1 for t in frame if 15.0 <= t <= 25.0)
    result = detect_threat(frame)
    status = "PASS" if result == expected else "FAIL"
    if status == "PASS":
        passed += 1
    print(f"{name:25} | body={body:3d} cold={cold:3d} | Expected: {expected:17} | Got: {result:17} | {status}")

print("=" * 78)
print(f"Passed: {passed} / {len(tests)}")
print("=" * 78)