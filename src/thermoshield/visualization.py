"""Visualisation utilities for synthetic thermal frames.

save_heatmap writes a single PNG using matplotlib's 'inferno' colormap.
The colormap is chosen to make hot and cold regions visually distinct;
it is not calibrated to a fixed temperature scale.
"""

import matplotlib.pyplot as plt

from .configuration import FRAME_WIDTH, FRAME_HEIGHT

__all__ = ["save_heatmap"]


def save_heatmap(frame, path, dpi=100):
    """Render a 768-element frame as a 32x24 heatmap and save it to path."""
    grid = frame.reshape(FRAME_HEIGHT, FRAME_WIDTH)
    plt.imshow(grid, cmap="inferno")
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close()
