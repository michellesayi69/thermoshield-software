import numpy as np

# ==========================================
# THRESHOLDS
# ==========================================
BODY_TEMP_MIN = 28.0
COLD_SPOT_MAX = 21.0
COLD_SPOT_MIN = 15.0
MIN_BODY_PIXELS = 50
MIN_COLD_PIXELS = 8


def detect_threat(frame):
    """
    Takes 768 temperature values and returns:
    'CLEAR', 'PERSON', or 'WEAPON DETECTED'
    """
    body_pixels = 0
    cold_pixels = 0

    for temp in frame:
        if temp >= BODY_TEMP_MIN:
            body_pixels += 1
        if COLD_SPOT_MIN <= temp <= COLD_SPOT_MAX:
            cold_pixels += 1

    if body_pixels >= MIN_BODY_PIXELS and cold_pixels >= MIN_COLD_PIXELS:
        return "WEAPON DETECTED"
    elif body_pixels >= MIN_BODY_PIXELS:
        return "PERSON"
    else:
        return "CLEAR"


def detect_threat_with_confidence(frame):
    """
    Same logic but returns a confidence score.
    Confidence is based on how far above the thresholds we are.
    """
    body_pixels = 0
    cold_pixels = 0

    for temp in frame:
        if temp >= BODY_TEMP_MIN:
            body_pixels += 1
        if COLD_SPOT_MIN <= temp <= COLD_SPOT_MAX:
            cold_pixels += 1

    if body_pixels >= MIN_BODY_PIXELS and cold_pixels >= MIN_COLD_PIXELS:
        confidence = min(100, 70 + (cold_pixels - MIN_COLD_PIXELS) * 2)
        return "WEAPON DETECTED", confidence
    elif body_pixels >= MIN_BODY_PIXELS:
        confidence = min(100, 70 + (body_pixels - MIN_BODY_PIXELS))
        return "PERSON", confidence
    else:
        confidence = 100
        return "CLEAR", confidence