"""Detection and frame constants for ThermoShield.

All temperatures are in degrees Celsius. Frame dimensions match the
MLX90640 native resolution (32 x 24 = 768 pixels).

These values are currently module-level constants. Extracting them into
a config object (so scenarios can be compared side-by-side without
editing code) is planned as future work — see README, Future work.
"""

# Frame geometry (MLX90640 native resolution).
FRAME_WIDTH = 32
FRAME_HEIGHT = 24
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT  # 768

# Detection thresholds. Chosen for ~22 °C indoor ambient, not calibrated.
BODY_TEMP_MIN = 28.0
COLD_SPOT_MIN = 15.0
COLD_SPOT_MAX = 21.0
MIN_BODY_PIXELS = 50
MIN_COLD_PIXELS = 8
