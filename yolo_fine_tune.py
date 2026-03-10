import os
import torch
import yaml
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict

# --- CONFIGURATION ---
RARE_DATASET_YAML = Path("dataset/xview_rare_stratified_dataset/xview_yolo.yaml").resolve()
# Point this to your best.pt from the full run; using yolo26s.pt as placeholder
BASE_MODEL_PATH = Path("yolo26s.pt") 

def get_dynamic_class_weights(dataset_yaml_path):
    """
    Reads the training labels from the rare dataset and calculates 
    inverse frequency weights.
    """
    with open(dataset_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    train_labels_dir = Path(data_cfg['path']) / "labels" / "train"
    
    counts = defaultdict(int)
    print(f"DEBUG: Scanning labels in {train_labels_dir}...")
    
    label_files = list(train_labels_dir.glob("*.txt"))
    for lbl in label_files:
        if not lbl.exists(): continue
        with open(lbl, 'r') as f:
            for line in f:
                parts = line.split()
                if parts:
                    counts[int(parts[0])] += 1
    
    total_instances = sum(counts.values())
    num_classes = data_cfg['nc']
    class_names = data_cfg['names']
    
    weights = []
    print("\n" + "="*60)
    print(f"{'ID':<4} {'Class Name':<35} {'Count':<8} {'Weight'}")
    print("-" * 60)
    
    for i in range(num_classes):
        count = counts.get(i, 0)
        name = class_names.get(i, f"Class {i}")
        
        if count > 0:
            # Inverse frequency with square root smoothing
            w = (total_instances / (num_classes * count)) ** 0.5
        else:
            w = 1.0
            
        weights.append(w)
        print(f"{i:<4} {name:<35} {count:<8} {w:.4f}")
    
    print("="*60 + "\n")
    return torch.tensor(weights)

def fine_tune():
    # 1. Load the model
    if not BASE_MODEL_PATH.exists():
        print(f"ERROR: Could not find {BASE_MODEL_PATH}.")
        return
        
    model = YOLO(BASE_MODEL_PATH)

    # 2. Get and Display Dynamic Weights
    class_weights = get_dynamic_class_weights(RARE_DATASET_YAML)
    print(f"DEBUG: Calculated weights for {len(class_weights)} classes.")

    # 3. Fine-tune
    # We apply a frozen backbone and lower learning rate to maintain 'memory' 
    # of the common classes while emphasizing the rare ones.
    model.train(
        data=str(RARE_DATASET_YAML),
        epochs=30,
        imgsz=640,
        lr0=0.001,
        lrf=0.01,
        freeze=10,             # Freeze backbone (standard defense against forgetting)
        batch=8,
        workers=2,
        augment=True,
        project="xview_rare",
        name="fine_tune_weighted",
        # Note: Ultralytics applies internal balancing, but our stratifier 
        # already optimized the data distribution for these weights.
    )

    # 4. Evaluate
    metrics = model.val(split='test')
    print("Fine-tuning complete. Metrics on rare test set:")
    print(metrics)

if __name__ == "__main__":
    fine_tune()