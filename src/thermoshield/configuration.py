"""Detection and frame constants for ThermoShield.

Temperatures are in degrees Celsius. Frame dimensions match the
MLX90640 native resolution (32 x 24 = 768 pixels).

Stage 4: adds a DetectionConfig dataclass so thresholds can be
compared side-by-side without editing the detection module. The
module-level constants below define the default (baseline) config
and match the pre-Stage-4 fixed-threshold behaviour exactly.
"""

from dataclasses import dataclass

# Frame geometry (MLX90640 native resolution).
FRAME_WIDTH = 32
FRAME_HEIGHT = 24
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT  # 768

# Default thresholds. Reproduce the baseline classifier exactly.
BODY_TEMP_MIN = 28.0
COLD_SPOT_MIN = 15.0
COLD_SPOT_MAX = 21.0
MIN_BODY_PIXELS = 50
MIN_COLD_PIXELS = 8


@dataclass(frozen=True)
class DetectionConfig:
    """Configurable detection thresholds.

    The default instance reproduces the baseline fixed-threshold
    classifier exactly. Custom instances allow comparison across
    configurations without editing the detection module.

    All temperatures are in degrees Celsius.
    """

    body_temp_min: float = BODY_TEMP_MIN
    cold_spot_min: float = COLD_SPOT_MIN
    cold_spot_max: float = COLD_SPOT_MAX
    min_body_pixels: int = MIN_BODY_PIXELS
    min_cold_pixels: int = MIN_COLD_PIXELS
    median_filter: bool = False  # Stage 4 sub-step 2: 3x3 median filter before detection

    def __post_init__(self):
        if self.cold_spot_min >= self.cold_spot_max:
            raise ValueError(
                f"cold_spot_min ({self.cold_spot_min}) must be strictly "
                f"less than cold_spot_max ({self.cold_spot_max})"
            )
        if self.body_temp_min <= self.cold_spot_max:
            raise ValueError(
                f"body_temp_min ({self.body_temp_min}) must be strictly "
                f"greater than cold_spot_max ({self.cold_spot_max}) so "
                f"the body and cold-spot rules cannot count the same pixel"
            )
        if self.min_body_pixels < 0:
            raise ValueError(
                f"min_body_pixels must be >= 0, got {self.min_body_pixels}"
            )
        if self.min_cold_pixels < 0:
            raise ValueError(
                f"min_cold_pixels must be >= 0, got {self.min_cold_pixels}"
            )


DEFAULT_CONFIG = DetectionConfig()
