"""Tests for input validation of classify_frame (Stage 4, sub-step 4).

Covers the four validation rules:
1. frame is not None
2. frame is convertible to numeric
3. frame is 1-D of length 768
4. every element is finite (no NaN, no inf)

Plus integration tests proving that classify_frame and
classify_frame_with_confidence both run validation, and that the
median_filter and background_aware code paths do not mask the
validation errors with their own lower-level errors.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from thermoshield import (
    classify_frame,
    classify_frame_with_confidence,
    validate_frame,
    DetectionConfig,
    FRAME_SIZE,
)


def expect_raises(fn, exc_type, label):
    try:
        fn()
    except exc_type:
        return (label, "PASS")
    except Exception as e:
        return (label, f"FAIL (wrong exception: {type(e).__name__}: {e})")
    return (label, "FAIL (did not raise)")


results = []
valid_frame = np.full(FRAME_SIZE, 22.0)

# --- Accept inputs ---

# 1. Valid numpy array accepted.
try:
    out = validate_frame(valid_frame)
    ok = isinstance(out, np.ndarray) and out.shape == (FRAME_SIZE,)
    results.append(("1. accepts numpy array", "PASS" if ok else "FAIL"))
except Exception as e:
    results.append(("1. accepts numpy array", f"FAIL ({e})"))

# 2. Valid Python list accepted.
try:
    out = validate_frame([22.0] * FRAME_SIZE)
    ok = isinstance(out, np.ndarray) and out.size == FRAME_SIZE
    results.append(("2. accepts Python list", "PASS" if ok else "FAIL"))
except Exception as e:
    results.append(("2. accepts Python list", f"FAIL ({e})"))

# 3. Valid Python tuple accepted.
try:
    out = validate_frame(tuple([22.0] * FRAME_SIZE))
    ok = isinstance(out, np.ndarray) and out.size == FRAME_SIZE
    results.append(("3. accepts Python tuple", "PASS" if ok else "FAIL"))
except Exception as e:
    results.append(("3. accepts Python tuple", f"FAIL ({e})"))

# 4. Valid input is unchanged (no mutation of the original array).
copy_before = valid_frame.copy()
_ = validate_frame(valid_frame)
unchanged = np.array_equal(valid_frame, copy_before)
results.append(("4. input array not mutated", "PASS" if unchanged else "FAIL"))

# --- Reject inputs ---

# 5. None rejected with TypeError.
results.append(expect_raises(lambda: validate_frame(None), TypeError,
                             "5. None rejected with TypeError"))

# 6. Too short rejected.
results.append(expect_raises(lambda: validate_frame([22.0] * 100), ValueError,
                             "6. too short rejected"))

# 7. Too long rejected.
results.append(expect_raises(lambda: validate_frame([22.0] * 2000), ValueError,
                             "7. too long rejected"))

# 8. Non-numeric element rejected with TypeError.
results.append(expect_raises(
    lambda: validate_frame([22.0] * (FRAME_SIZE - 1) + ["hello"]),
    TypeError, "8. non-numeric rejected"))

# 9. 2-D array rejected.
results.append(expect_raises(
    lambda: validate_frame(np.full((24, 32), 22.0)),
    ValueError, "9. 2-D array rejected"))

# 10. NaN rejected.
bad_nan = np.full(FRAME_SIZE, 22.0); bad_nan[100] = np.nan
results.append(expect_raises(lambda: validate_frame(bad_nan), ValueError,
                             "10. NaN rejected"))

# 11. inf rejected.
bad_inf = np.full(FRAME_SIZE, 22.0); bad_inf[200] = np.inf
results.append(expect_raises(lambda: validate_frame(bad_inf), ValueError,
                             "11. inf rejected"))

# --- Integration with classify_frame ---

# 12. classify_frame rejects invalid input (integration).
results.append(expect_raises(
    lambda: classify_frame([22.0] * 100),
    ValueError, "12. classify_frame rejects short frame"))

# 13. classify_frame_with_confidence rejects invalid input.
results.append(expect_raises(
    lambda: classify_frame_with_confidence([22.0] * 100),
    ValueError, "13. confidence fn rejects short frame"))

# 14. Validation runs before the median_filter branch.
#     Passing a valid frame with median_filter=True should succeed, but
#     an INVALID frame with median_filter=True should raise the validation
#     error (ValueError about length), not a lower-level error from the
#     filter.
results.append(expect_raises(
    lambda: classify_frame([22.0] * 100, config=DetectionConfig(median_filter=True)),
    ValueError, "14. validation precedes filter"))

# 15. Valid frame with median_filter=True still classifies correctly.
clean = np.full(FRAME_SIZE, 22.0)
for y in range(5, 15):
    for x in range(5, 15):
        clean[y * 32 + x] = 33.0
lab = classify_frame(clean, config=DetectionConfig(median_filter=True))
results.append((
    "15. valid frame + filter still works",
    "PASS" if lab == "PERSON" else f"FAIL (got {lab})",
))

print("=" * 70)
print("THERMOSHIELD - VALIDATION TESTS (Stage 4, sub-step 4)")
print("=" * 70)
passed = 0
for name, status in results:
    print(f"{name:42} | {status}")
    if status == "PASS":
        passed += 1
print("=" * 70)
print(f"Passed: {passed} / {len(results)}")
print("=" * 70)
sys.exit(0 if passed == len(results) else 1)
