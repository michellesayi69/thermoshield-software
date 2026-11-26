"""Threshold-based thermal anomaly classifier.

This is the baseline algorithm described in the README. It counts pixels
above the body-temperature threshold and pixels inside the cold-spot
temperature band, then applies two minimum-count rules to assign one of
three labels: CLEAR, PERSON, or THERMAL ANOMALY.

No machine learning is used. The whole algorithm is visible here.

Known limitation (documented, not fixed in this version): the
confidence score for CLEAR is always 100, regardless of how close the
frame was to triggering a PERSON or ANOMALY label. See README,
Future work.
"""

from .configuration import (
    BODY_TEMP_MIN,
    COLD_SPOT_MIN,
    COLD_SPOT_MAX,
    MIN_BODY_PIXELS,
    MIN_COLD_PIXELS,
)

__all__ = ["classify_frame", "classify_frame_with_confidence"]


def _count_pixels(frame):
    """Return (body_pixels, cold_pixels) counted against current thresholds."""
    body_pixels = 0
    cold_pixels = 0
    for temp in frame:
        if temp >= BODY_TEMP_MIN:
            body_pixels += 1
        if COLD_SPOT_MIN <= temp <= COLD_SPOT_MAX:
            cold_pixels += 1
    return body_pixels, cold_pixels


def classify_frame(frame):
    """Classify a thermal frame (iterable of 768 temperature values).

    Returns one of: 'CLEAR', 'PERSON', 'THERMAL ANOMALY'.
    """
    body_pixels, cold_pixels = _count_pixels(frame)

    if body_pixels >= MIN_BODY_PIXELS and cold_pixels >= MIN_COLD_PIXELS:
        return "THERMAL ANOMALY"
    elif body_pixels >= MIN_BODY_PIXELS:
        return "PERSON"
    else:
        return "CLEAR"


def classify_frame_with_confidence(frame):
    """Classify a thermal frame and also return a heuristic confidence (0-100).

    The confidence formula is documented here so that a reviewer can see
    the exact rule rather than being presented with an opaque number.
    """
    body_pixels, cold_pixels = _count_pixels(frame)

    if body_pixels >= MIN_BODY_PIXELS and cold_pixels >= MIN_COLD_PIXELS:
        confidence = min(100, 70 + (cold_pixels - MIN_COLD_PIXELS) * 2)
        return "THERMAL ANOMALY", confidence
    elif body_pixels >= MIN_BODY_PIXELS:
        confidence = min(100, 70 + (body_pixels - MIN_BODY_PIXELS))
        return "PERSON", confidence
    else:
        confidence = 100
        return "CLEAR", confidence
