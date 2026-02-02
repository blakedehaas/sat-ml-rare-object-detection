from ultralytics import YOLO
from pathlib import Path
import os

# Change to dataset directory
DATASET_ROOT = Path("dataset/xview_cleaned_dataset").resolve()
os.chdir(DATASET_ROOT)
print(f"Working directory: {os.getcwd()}")

# Now paths are relative to dataset root
DATA_YAML = Path("YOLO_cfg/xview_yolo.yaml")

# Delete cache
# for cache_file in Path(".").glob("*.cache"):
#     cache_file.unlink()
#     print(f"Deleted cache: {cache_file}")

# Load model (adjust path since we changed directory)

model = YOLO("yolo26n.pt") 

# Train
#results = model.train(data=str(DATA_YAML), epochs=50, imgsz=640)
results = model.train(data=str(DATA_YAML), epochs=5, imgsz=416)

# Evaluate
metrics = model.val(data=str(DATA_YAML), split="test")

print(metrics)
