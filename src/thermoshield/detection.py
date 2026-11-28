"""Threshold-based thermal anomaly classifier.

The baseline algorithm from the README. No machine learning.

Stage 4: accepts an optional DetectionConfig. When config is None,
DEFAULT_CONFIG is used, which reproduces the pre-Stage-4 behaviour
exactly. Passing a custom config allows side-by-side comparison.
"""

from .configuration import DetectionConfig, DEFAULT_CONFIG

__all__ = ["classify_frame", "classify_frame_with_confidence"]


def _count_pixels(frame, config):
    """Return (body_pixels, cold_pixels) counted against config thresholds."""
    body_pixels = 0
    cold_pixels = 0
    for temp in frame:
        if temp >= config.body_temp_min:
            body_pixels += 1
        if config.cold_spot_min <= temp <= config.cold_spot_max:
            cold_pixels += 1
    return body_pixels, cold_pixels


def classify_frame(frame, config=None):
    """Classify a thermal frame.

    Parameters
    ----------
    frame : iterable of float
        768 temperature values (32 x 24).
    config : DetectionConfig, optional
        Threshold configuration. Defaults to DEFAULT_CONFIG.

    Returns one of: 'CLEAR', 'PERSON', 'THERMAL ANOMALY'.
    """
    if config is None:
        config = DEFAULT_CONFIG
    body_pixels, cold_pixels = _count_pixels(frame, config)
    if body_pixels >= config.min_body_pixels and cold_pixels >= config.min_cold_pixels:
        return "THERMAL ANOMALY"
    elif body_pixels >= config.min_body_pixels:
        return "PERSON"
    else:
        return "CLEAR"


def classify_frame_with_confidence(frame, config=None):
    """Classify a thermal frame and also return a heuristic confidence (0-100).

    Parameters and return-value contract are identical to classify_frame,
    except the label is returned as a tuple (label, confidence).
    """
    if config is None:
        config = DEFAULT_CONFIG
    body_pixels, cold_pixels = _count_pixels(frame, config)
    if body_pixels >= config.min_body_pixels and cold_pixels >= config.min_cold_pixels:
        confidence = min(100, 70 + (cold_pixels - config.min_cold_pixels) * 2)
        return "THERMAL ANOMALY", confidence
    elif body_pixels >= config.min_body_pixels:
        confidence = min(100, 70 + (body_pixels - config.min_body_pixels))
        return "PERSON", confidence
    else:
        return "CLEAR", 100
