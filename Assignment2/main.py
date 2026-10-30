import os
from pathlib import Path

dataset_path = Path(__file__).resolve().parent / "age_gap_dataset"

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

count = 0

for root, dirs, files in os.walk(str(dataset_path)):
    for file in files:
        if file.lower().endswith(image_extensions):
            count += 1

print("Total images:", count)
