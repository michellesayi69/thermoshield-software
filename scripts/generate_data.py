import numpy as np
import matplotlib.pyplot as plt
import os

IMAGES_PER_CLASS = 60
GRID_WIDTH = 32
GRID_HEIGHT = 24
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

os.makedirs("data/weapon", exist_ok=True)
os.makedirs("data/person", exist_ok=True)
os.makedirs("data/clear", exist_ok=True)

def save_image(frame, path):
    grid = frame.reshape(GRID_HEIGHT, GRID_WIDTH)
    plt.imshow(grid, cmap='inferno')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_clear():
    return np.random.normal(22.0, 0.5, GRID_WIDTH * GRID_HEIGHT)

def generate_person():
    frame = np.random.normal(22.0, 0.5, GRID_WIDTH * GRID_HEIGHT)
    body_size = np.random.randint(80, 120)
    for i in np.random.choice(GRID_WIDTH * GRID_HEIGHT, body_size, replace=False):
        frame[i] = np.random.normal(33.0, 0.8)
    return frame

def generate_weapon():
    frame = generate_person()
    cold_size = np.random.randint(10, 30)
    for i in np.random.choice(GRID_WIDTH * GRID_HEIGHT, cold_size, replace=False):
        frame[i] = np.random.normal(20.0, 0.8)
    return frame

for i in range(IMAGES_PER_CLASS):
    save_image(generate_clear(), f"data/clear/clear_{i+1:02d}.png")
    save_image(generate_person(), f"data/person/person_{i+1:02d}.png")
    save_image(generate_weapon(), f"data/weapon/weapon_{i+1:02d}.png")

print(f"Generated {IMAGES_PER_CLASS * 3} images successfully.")