"""Frame validation for the public classifier API.

Stage 4, sub-step 4: classify_frame and classify_frame_with_confidence
accept a frame argument from external callers. This module provides a
single validator that runs before the detection logic so that the
algorithm never operates on an input it was not designed for.

The validator enforces:
- frame is not None
- frame is convertible to a 1-D numeric numpy array
- frame has exactly FRAME_SIZE (768) elements
- every element is finite (no NaN, no inf)

A frame that fails validation cannot be silently misclassified. The
error message identifies exactly which rule was violated and where.
"""
import numpy as np

from .configuration import FRAME_SIZE

__all__ = ["validate_frame"]


def validate_frame(frame):
    """Validate a thermal frame and return a normalised numpy array.

    Args:
        frame: any iterable of numeric values, or a numpy array.

    Returns:
        A 1-D numpy array (float64) of exactly 768 finite values.

    Raises:
        TypeError: if frame is None, or is not convertible to numeric.
        ValueError: if frame is not 1-D, has wrong length, or contains
                    a non-finite value (NaN or +/-inf).
    """
    if frame is None:
        raise TypeError("frame must not be None")

    try:
        arr = np.asarray(frame, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"frame must be an iterable of numbers; conversion failed: {exc}"
        ) from exc

    if arr.ndim != 1:
        raise ValueError(
            f"frame must be 1-dimensional; got shape {arr.shape}. "
            f"Pass a flat sequence of {FRAME_SIZE} values."
        )
    if arr.size != FRAME_SIZE:
        raise ValueError(
            f"frame must contain {FRAME_SIZE} values (32 x 24); got {arr.size}"
        )

    non_finite = ~np.isfinite(arr)
    if non_finite.any():
        first_bad = int(np.argmax(non_finite))
        raise ValueError(
            f"frame contains a non-finite value at index {first_bad}: "
            f"{arr[first_bad]}"
        )

    return arr
