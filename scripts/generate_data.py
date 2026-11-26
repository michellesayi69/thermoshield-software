"""Generate 180 synthetic thermal frames (60 per scenario) into data/.

This is a thin entry point. The actual generation logic lives in
src/thermoshield/data_generation.py; this script only handles the
file-system layout and calls the library functions.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from thermoshield import seed_rng, generate_clear, generate_person, generate_anomaly
from thermoshield.visualization import save_heatmap

IMAGES_PER_CLASS = 60


def main():
    if os.path.exists("data/weapon"):
        print("Note: data/weapon/ exists from an earlier run.")
        print("      This version writes to data/anomaly/. You may remove")
        print("      the old folder with:  rm -rf data/weapon\n")

    for sub in ("clear", "person", "anomaly"):
        os.makedirs(f"data/{sub}", exist_ok=True)

    seed_rng()
    for i in range(IMAGES_PER_CLASS):
        save_heatmap(generate_clear(), f"data/clear/clear_{i+1:02d}.png")
        save_heatmap(generate_person(), f"data/person/person_{i+1:02d}.png")
        save_heatmap(generate_anomaly(), f"data/anomaly/anomaly_{i+1:02d}.png")

    print(f"Generated {IMAGES_PER_CLASS * 3} images successfully.")


if __name__ == "__main__":
    main()
