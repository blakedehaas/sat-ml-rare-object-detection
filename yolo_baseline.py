from ultralytics import YOLO
from pathlib import Path
import os

# Change to dataset directory
DATASET_ROOT = Path("dataset/xview_stratified_dataset").resolve()
os.chdir(DATASET_ROOT)
print(f"Working directory: {os.getcwd()}")

# Now paths are relative to dataset root
DATA_YAML = Path("dataset/xview_stratified_dataset/xview_yolo.yaml")

# Load model
model = YOLO("yolo26s.pt") 

# Train
results = model.train(data=str(DATA_YAML), epochs=50, imgsz=640)

# Evaluate
metrics = model.val(data=str(DATA_YAML), split="test")

print(metrics)
