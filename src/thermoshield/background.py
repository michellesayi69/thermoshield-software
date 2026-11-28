"""Background-temperature estimation for thermal frames.

Stage 4, sub-step 3: background-aware detection needs an estimate of
the prevailing scene temperature so that body and cold-spot rules can
be expressed as deltas relative to that estimate rather than as
absolute temperatures.

The estimator is deliberately simple: the median of all 768 pixel
values. The median is robust to a minority of extreme pixels (the body
and cold-spot regions typically cover well under half the frame), so
it tracks the ambient temperature rather than being pulled toward the
features being detected.
"""
import numpy as np

__all__ = ["estimate_background"]


def estimate_background(frame):
    """Return the estimated background temperature of a thermal frame.

    Uses the median of all pixel values. See module docstring for
    reasoning.

    Args:
        frame: iterable of temperature values (Celsius)

    Returns:
        float, the estimated ambient temperature
    """
    return float(np.median(np.asarray(frame, dtype=float)))
