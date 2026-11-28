"""ThermoShield — thermal anomaly detection simulation.

A small, honest, threshold-based classifier for synthetic 32x24 thermal
frames. No machine learning, no trained model. See README for scope,
methodology, and limitations.

Public API
----------
Detector
    classify_frame(frame)               -> str
    classify_frame_with_confidence(frame) -> (str, int)

Data generation
    seed_rng(seed=42)
    generate_clear()   -> np.ndarray
    generate_person()  -> np.ndarray
    generate_anomaly() -> np.ndarray

Configuration
    BODY_TEMP_MIN, COLD_SPOT_MIN, COLD_SPOT_MAX,
    MIN_BODY_PIXELS, MIN_COLD_PIXELS,
    FRAME_WIDTH, FRAME_HEIGHT, FRAME_SIZE,
    DetectionConfig, DEFAULT_CONFIG
"""

from .configuration import (
    BODY_TEMP_MIN,
    COLD_SPOT_MIN,
    COLD_SPOT_MAX,
    MIN_BODY_PIXELS,
    MIN_COLD_PIXELS,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FRAME_SIZE,
    DetectionConfig,
    DEFAULT_CONFIG,
)
from .detection import (
    classify_frame,
    classify_frame_with_confidence,
)
from .data_generation import (
    seed_rng,
    generate_clear,
    generate_person,
    generate_anomaly,
)
from .filtering import median_filter
from .background import estimate_background
from .validation import validate_frame

__version__ = "0.1.0"

__all__ = [
    "classify_frame",
    "classify_frame_with_confidence",
    "seed_rng",
    "generate_clear",
    "generate_person",
    "generate_anomaly",
    "BODY_TEMP_MIN",
    "COLD_SPOT_MIN",
    "COLD_SPOT_MAX",
    "MIN_BODY_PIXELS",
    "MIN_COLD_PIXELS",
    "FRAME_WIDTH",
    "FRAME_HEIGHT",
    "FRAME_SIZE",
    "DetectionConfig",
    "DEFAULT_CONFIG",
    "median_filter",
    "estimate_background",
    "validate_frame",
]
