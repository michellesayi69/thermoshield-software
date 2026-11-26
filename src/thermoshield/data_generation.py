"""Synthetic thermal frame generation.

All frames are synthetic — they are NOT real MLX90640 sensor data. The
generator is deterministic when seeded and exists only to provide
controlled inputs for testing the detection algorithm and producing
demo images. See README, Limitations.
"""

import numpy as np

from .configuration import FRAME_SIZE

__all__ = [
    "generate_clear",
    "generate_person",
    "generate_anomaly",
    "seed_rng",
]

DEFAULT_SEED = 42

AMBIENT_MEAN = 22.0
AMBIENT_STD = 0.5
BODY_MEAN = 33.0
BODY_STD = 0.8
COLD_MEAN = 20.0
COLD_STD = 0.8

BODY_PIXELS_RANGE = (80, 120)
COLD_PIXELS_RANGE = (10, 30)


def seed_rng(seed=DEFAULT_SEED):
    """Seed the global numpy RNG so generated frames are reproducible."""
    np.random.seed(seed)


def generate_clear():
    """Synthetic empty scene: all pixels near ambient (~22 °C)."""
    return np.random.normal(AMBIENT_MEAN, AMBIENT_STD, FRAME_SIZE)


def generate_person():
    """Synthetic scene: ambient background plus a warm 'body' region."""
    frame = np.random.normal(AMBIENT_MEAN, AMBIENT_STD, FRAME_SIZE)
    body_size = np.random.randint(*BODY_PIXELS_RANGE)
    for i in np.random.choice(FRAME_SIZE, body_size, replace=False):
        frame[i] = np.random.normal(BODY_MEAN, BODY_STD)
    return frame


def generate_anomaly():
    """Synthetic scene: a person-like frame plus a distinct cold region.

    This is the scenario the detector labels THERMAL ANOMALY. The
    'anomaly' is a cold patch superimposed on a person-like frame. It
    does not correspond to any specific object and is not a weapon or
    threat indicator. See README, Detection methodology.
    """
    frame = generate_person()
    cold_size = np.random.randint(*COLD_PIXELS_RANGE)
    for i in np.random.choice(FRAME_SIZE, cold_size, replace=False):
        frame[i] = np.random.normal(COLD_MEAN, COLD_STD)
    return frame
