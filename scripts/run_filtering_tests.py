"""Tests for the median filter (Stage 4, sub-step 2).

Proves: (a) the filter is a correct median filter, (b) it removes
isolated noise pixels, (c) it preserves contiguous regions, and (d)
when enabled via DetectionConfig it eliminates false positives that
the unfiltered baseline produces on noisy input.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from thermoshield import (
    classify_frame,
    DetectionConfig,
    median_filter,
    FRAME_SIZE,
)


results = []

# 1. Identity on a constant frame.
flat = np.full(FRAME_SIZE, 22.0)
filtered = median_filter(flat)
results.append((
    "1. identity on flat frame",
    "PASS" if np.allclose(filtered, flat) else "FAIL",
))

# 2. Salt-and-pepper: isolated extremes removed.
frame = np.full(FRAME_SIZE, 22.0)
frame[0] = 33.0
frame[400] = 19.0
filtered = median_filter(frame)
isolated_gone = (filtered[0] != 33.0) and (filtered[400] != 19.0)
results.append((
    "2. isolated extremes removed",
    "PASS" if isolated_gone else f"FAIL (0={filtered[0]}, 400={filtered[400]})",
))

# 3. Contiguous body region preserved.
frame = np.full(FRAME_SIZE, 22.0)
# 10x10 contiguous body region at rows 10-19, cols 10-19
for y in range(10, 20):
    for x in range(10, 20):
        frame[y * 32 + x] = 33.0
filtered = median_filter(frame).reshape(24, 32)
# Interior of region (rows 12-17, cols 12-17) should remain 33
interior_kept = all(filtered[y, x] == 33.0 for y in range(12, 18) for x in range(12, 18))
results.append((
    "3. contiguous region interior kept",
    "PASS" if interior_kept else "FAIL",
))

# 4. Filter does not create false positives on clean frames.
clean = np.full(FRAME_SIZE, 22.0)
for y in range(10, 20):
    for x in range(10, 20):
        clean[y * 32 + x] = 33.0
lab_off = classify_frame(clean)
lab_on = classify_frame(clean, config=DetectionConfig(median_filter=True))
results.append((
    "4. filter preserves clean-frame label",
    "PASS" if lab_off == lab_on == "PERSON" else f"FAIL ({lab_off} -> {lab_on})",
))

# 5. KEY TEST: filter eliminates noise-induced false positive.
# Construct a frame with:
#   - 10x10 body region (100 body pixels, survives filter as ~64)
#   - 20 isolated single cold pixels scattered far from body
# Unfiltered: body=100, cold=20 -> THERMAL ANOMALY (FALSE POSITIVE)
# Filtered:   body~64, cold=0  -> PERSON (correct)
noisy = np.full(FRAME_SIZE, 22.0)
for y in range(5, 15):
    for x in range(5, 15):
        noisy[y * 32 + x] = 33.0
noise_positions = [0, 33, 66, 99, 200, 233, 266, 300, 400, 433,
                   500, 533, 566, 600, 633, 666, 700, 733, 760, 767]
for p in noise_positions:
    noisy[p] = 19.0
lab_off = classify_frame(noisy)
lab_on = classify_frame(noisy, config=DetectionConfig(median_filter=True))
results.append((
    "5. filter fixes noise false positive",
    "PASS" if lab_off == "THERMAL ANOMALY" and lab_on == "PERSON"
    else f"FAIL (off={lab_off}, on={lab_on})",
))

# 6. Filter preserves a real anomaly (large contiguous cold region).
real_anomaly = np.full(FRAME_SIZE, 22.0)
for y in range(5, 15):
    for x in range(5, 15):
        real_anomaly[y * 32 + x] = 33.0  # body
for y in range(12, 16):
    for x in range(20, 24):
        real_anomaly[y * 32 + x] = 19.0  # 4x4 cold region = 16 pixels
lab_off = classify_frame(real_anomaly)
lab_on = classify_frame(real_anomaly, config=DetectionConfig(median_filter=True))
results.append((
    "6. filter preserves real anomaly",
    "PASS" if lab_off == lab_on == "THERMAL ANOMALY" else f"FAIL ({lab_off} -> {lab_on})",
))

# 7. Invalid size rejected.
try:
    median_filter(clean, size=2)
    results.append(("7. invalid size rejected", "FAIL (no exception)"))
except ValueError:
    results.append(("7. invalid size rejected", "PASS"))

# 8. Invalid frame size rejected.
try:
    median_filter(np.zeros(100))
    results.append(("8. invalid frame size rejected", "FAIL (no exception)"))
except ValueError:
    results.append(("8. invalid frame size rejected", "PASS"))

print("=" * 70)
print("THERMOSHIELD - FILTERING TESTS (Stage 4, sub-step 2)")
print("=" * 70)
passed = 0
for name, status in results:
    print(f"{name:36} | {status}")
    if status == "PASS":
        passed += 1
print("=" * 70)
print(f"Passed: {passed} / {len(results)}")
print("=" * 70)
sys.exit(0 if passed == len(results) else 1)
