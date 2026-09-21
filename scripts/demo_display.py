import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from detect_threat import detect_threat, detect_threat_with_confidence


def save_image(frame, path):
    grid = frame.reshape(24, 32)
    plt.imshow(grid, cmap='inferno')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def generate_clear():
    return np.random.normal(22.0, 0.5, 768)


def generate_person():
    frame = np.random.normal(22.0, 0.5, 768)
    body_size = np.random.randint(80, 120)
    for i in np.random.choice(768, body_size, replace=False):
        frame[i] = np.random.normal(33.0, 0.8)
    return frame


def generate_weapon():
    frame = generate_person()
    cold_size = np.random.randint(10, 30)
    for i in np.random.choice(768, cold_size, replace=False):
        frame[i] = np.random.normal(20.0, 0.8)
    return frame


def show_demo(frame, title):
    result, confidence = detect_threat_with_confidence(frame)
    grid = frame.reshape(24, 32)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: heatmap
    im = ax1.imshow(grid, cmap='inferno')
    ax1.set_title("Thermal Input (32x24)", fontsize=14)
    ax1.axis('off')
    plt.colorbar(im, ax=ax1, label='Temperature (°C)')

    # Right: result banner
    if result == "WEAPON DETECTED":
        color = '#ff4444'
    elif result == "PERSON":
        color = '#ffaa00'
    else:
        color = '#44cc44'

    ax2.text(0.5, 0.6, result, fontsize=22, ha='center', va='center',
             fontweight='bold',
             bbox=dict(facecolor=color, alpha=0.4, pad=20, edgecolor='black'))
    ax2.text(0.5, 0.3, f"Confidence: {confidence}%", fontsize=14,
             ha='center', va='center')
    ax2.set_title("Detection Result", fontsize=14)
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.suptitle(f"ThermoShield Demo — {title}", fontsize=16)
    plt.tight_layout()

    filename = f"results/demo_{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# Generate demo images
os.makedirs("results", exist_ok=True)
np.random.seed(1)

show_demo(generate_clear(), "Clear Room")
show_demo(generate_person(), "Person")
show_demo(generate_weapon(), "Weapon Detected")

print("Demo complete. Check the results/ folder.")