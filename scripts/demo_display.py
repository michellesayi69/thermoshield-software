"""Render demo heatmaps for one frame of each scenario.

Writes three PNGs to results/. The generator and detection logic live
in the thermoshield package; this script only handles layout and
matplotlib figure creation. The duplication of generator functions
that existed in the original demo_display.py is gone.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
from thermoshield import (
    classify_frame_with_confidence,
    seed_rng,
    generate_clear,
    generate_person,
    generate_anomaly,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)


def show_demo(frame, title):
    result, confidence = classify_frame_with_confidence(frame)
    grid = frame.reshape(FRAME_HEIGHT, FRAME_WIDTH)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im = ax1.imshow(grid, cmap="inferno")
    ax1.set_title("Thermal Input (32x24)", fontsize=14)
    ax1.axis("off")
    plt.colorbar(im, ax=ax1, label="Temperature (C)")

    if result == "THERMAL ANOMALY":
        color = "#ff4444"
    elif result == "PERSON":
        color = "#ffaa00"
    else:
        color = "#44cc44"

    ax2.text(0.5, 0.6, result, fontsize=22, ha="center", va="center",
             fontweight="bold",
             bbox=dict(facecolor=color, alpha=0.4, pad=20, edgecolor="black"))
    ax2.text(0.5, 0.3, f"Confidence: {confidence}%", fontsize=14,
             ha="center", va="center")
    ax2.set_title("Detection Result", fontsize=14)
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.suptitle(f"ThermoShield Demo - {title}", fontsize=16)
    plt.tight_layout()

    filename = f"results/demo_{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def main():
    os.makedirs("results", exist_ok=True)
    seed_rng(seed=1)  # Different seed than data_generation for fresh demo frames
    show_demo(generate_clear(), "Clear Room")
    show_demo(generate_person(), "Person")
    show_demo(generate_anomaly(), "Thermal Anomaly")
    print("Demo complete. Check the results/ folder.")


if __name__ == "__main__":
    main()
