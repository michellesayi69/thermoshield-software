"""Spatial filtering for thermal frames.

Stage 4, sub-step 2: a 3 x 3 median filter that removes isolated
noise pixels before detection. Implemented with numpy only (no scipy)
so that the operation is inspectable and the dependency list does not
grow. At 32x24 = 768 pixels the double loop is negligible.
"""
import numpy as np

from .configuration import FRAME_WIDTH, FRAME_HEIGHT

__all__ = ["median_filter"]


def median_filter(frame, size=3):
    """Apply a size x size median filter to a 768-element frame.

    Edge padding uses 'edge' mode (replicate border values) so the
    output preserves the 32x24 grid and 768-element length exactly.

    Args:
        frame: iterable of 768 temperature values (Celsius)
        size: filter size, must be odd (default 3)

    Returns:
        numpy.ndarray of 768 floats, same length as input

    Raises:
        ValueError: if size is not odd, or frame length is not 768
    """
    if size % 2 != 1:
        raise ValueError(f"filter size must be odd, got {size}")

    arr = np.asarray(frame, dtype=float)
    if arr.size != FRAME_WIDTH * FRAME_HEIGHT:
        raise ValueError(
            f"expected {FRAME_WIDTH * FRAME_HEIGHT} values, got {arr.size}"
        )
    grid = arr.reshape(FRAME_HEIGHT, FRAME_WIDTH)

    pad = size // 2
    padded = np.pad(grid, pad, mode="edge")

    result = np.empty_like(grid)
    for y in range(FRAME_HEIGHT):
        for x in range(FRAME_WIDTH):
            window = padded[y:y + size, x:x + size]
            result[y, x] = np.median(window)

    return result.flatten()
