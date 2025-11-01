import os
import json
import random
import numpy as np
from datetime import datetime
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# ==== IMPORT YOUR GENERATOR FUNCTIONS ====
from all_interferences import *

# ===================================================
# STEP 1: Generate Synthetic Multi-Signal Interference Frame
# ===================================================

def generate_interference_frame():
    x = np.arange(0, 101, 1)
    y = np.arange(0, 21, 1)
    noise_grid = generate_gaussian_noise_grid(x, y, mean=10, variance=2)
    grid = noise_grid.copy()
    positions_log = []

    # Add mixed signals (simulate real-world coexistence)
    grid = add_wifi_clients(noise_grid,grid, x, y, n_clients=random.randint(1, 3), positions_log=positions_log)
    grid = add_zigbee_clients(noise_grid,grid, x, y, n_clients=random.randint(1, 5), positions_log=positions_log)
    grid = add_bluetooth_clients(noise_grid,grid, x, y, n_clients=random.randint(5, 15), positions_log=positions_log)
    grid = add_cordless_phone_clients(noise_grid,grid, x, y, n_clients=random.randint(1, 3), positions_log=positions_log)

    grid -= grid.min()
    grid = (grid / grid.max()) * 255
    grid = np.clip(grid, 0, 255)

    # Save visualization
    os.makedirs("captures", exist_ok=True)
    img_path = f"captures/interference_{datetime.now().strftime('%H%M%S')}.png"
    Image.fromarray(grid.astype(np.uint8)).save(img_path)

    # Assign metadata per detected signal
    events = []
    for entry in positions_log:
        t = entry["type"]
        cx = entry["center_x"]
        cy = entry["center_y"]

        event = {
            "type": t,
            "confidence": round(random.uniform(0.85, 0.99), 2),  # classifier confidence
            "center_freq": round(np.interp(cy, [0, len(y)], [2400, 2483.5]), 2),
            "bandwidth_px": entry.get("bandwidth_px", 1),
            "amplitude": entry.get("amplitude", None),
            "duty_cycle": entry.get("duty_cycle", None),
            "noise_floor": entry.get("noise_floor", None),
            "airtime_cost": round(random.uniform(0.01, 0.8), 3),
        }
        events.append(event)

    frame_info = {
        "timestamp": datetime.now().isoformat(),
        "channel": random.choice([1, 6, 11]),
        "band": "2.4GHz",
        "events": events
    }

    print(f"Generated frame with {len(events)} signals")
    return img_path, frame_info


# ===================================================
# STEP 2: CNN Classifier (optional demo using pretrained ResNet18)
# ===================================================

class DummyInterferenceClassifier:
    """
    A placeholder classifier that assigns confidence to detected events.
    You can later replace this with a trained CNN model.
    """
    def predict(self, image_path, events):
        # Load image (just to simulate usage)
        img = Image.open(image_path).convert("RGB")
        _ = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])(img)
        # Add random confidence updates
        for ev in events:
            ev["confidence"] = round(min(1.0, ev["confidence"] + random.uniform(-0.05, 0.05)), 2)
        return events


# ===================================================
# STEP 3: Run the Simulation and Save Final Output
# ===================================================

def main():
    image_path, frame_info = generate_interference_frame()

    # Use CNN classifier (dummy demo)
    classifier = DummyInterferenceClassifier()
    frame_info["events"] = classifier.predict(image_path, frame_info["events"])

    # Save structured JSON
    os.makedirs("results", exist_ok=True)
    out_path = f"results/interference_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(frame_info, f, indent=4)

    print(f"Saved report to {out_path}")
    print(json.dumps(frame_info, indent=4))


if __name__ == "__main__":
    main()