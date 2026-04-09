import os
import torch
import yaml
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict

# --- CONFIGURATION ---
#RARE_DATASET_YAML = Path("dataset/xview_rare_stratified_dataset/xview_yolo.yaml").resolve()
RARE_DATASET_YAML = Path("/scratch/alpine/brne5584/sat-ml-rare-object-detection/dataset/xview_rare_stratified_dataset/xview_yolo.yaml")
# Point this to your best.pt from the full run; using yolo26s.pt as placeholder
# BASE_MODEL_PATH = Path("yolo26s.pt") 
BASE_MODEL_PATH = Path("/scratch/alpine/brne5584/sat-ml-rare-object-detection/best_baseline.pt")
CHECKPOINT_DIR = Path("/scratch/alpine/brne5584/sat-ml-rare-object-detection/runs/detect/xview_rare/fine_tune_weighted/weights")
TOTAL_EPOCHS = 30

def get_dynamic_class_weights(dataset_yaml_path):
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
            w = (total_instances / (num_classes * count)) ** 0.5
        else:
            w = 1.0
        weights.append(w)
        print(f"{i:<4} {name:<35} {count:<8} {w:.4f}")

    print("="*60 + "\n")
    return torch.tensor(weights)

def find_latest_checkpoint():
    """Find the latest epoch checkpoint if one exists."""
    if not CHECKPOINT_DIR.exists():
        return None, 0

    # Look for epoch checkpoints saved as epoch_N.pt
    checkpoints = list(CHECKPOINT_DIR.glob("epoch*.pt"))
    if not checkpoints:
        return None, 0

    # Extract epoch numbers and find the latest
    latest = max(checkpoints, key=lambda p: int(p.stem.replace("epoch", "")))
    epoch_num = int(latest.stem.replace("epoch", ""))
    print(f"Found checkpoint at epoch {epoch_num}: {latest}")
    return latest, epoch_num

def fine_tune():
    # 1. Check for existing checkpoint
    checkpoint, start_epoch = find_latest_checkpoint()

    if checkpoint:
        print(f"Resuming from epoch {start_epoch}...")
        model = YOLO(str(checkpoint))
    else:
        if not BASE_MODEL_PATH.exists():
            print(f"ERROR: Could not find {BASE_MODEL_PATH}.")
            return
        print("No checkpoint found, starting from base model...")
        model = YOLO(str(BASE_MODEL_PATH))

    remaining_epochs = TOTAL_EPOCHS - start_epoch
    if remaining_epochs <= 0:
        print("Training already complete!")
        return

    print(f"Training epochs {start_epoch + 1} to {TOTAL_EPOCHS} ({remaining_epochs} remaining)...")

    # 2. Get dynamic class weights
    class_weights = get_dynamic_class_weights(RARE_DATASET_YAML)
    print(f"DEBUG: Calculated weights for {len(class_weights)} classes.")

    # 3. Fine-tune
    model.train(
        data=str(RARE_DATASET_YAML),
        epochs=remaining_epochs,
        imgsz=640,
        lr0=0.001,
        lrf=0.01,
        freeze=10,
        batch=16,
        workers=4,
        augment=True,
        project="xview_rare",
        name="fine_tune_weighted",
        exist_ok=True,
        save_period=1,   # Save checkpoint every epoch
    )

    # 4. Save epoch checkpoint for resuming
    completed_epoch = start_epoch + remaining_epochs
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    latest_weights = CHECKPOINT_DIR / "last.pt"
    if latest_weights.exists():
        checkpoint_path = CHECKPOINT_DIR / f"epoch_{completed_epoch}.pt"
        import shutil
        shutil.copy2(latest_weights, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    # 5. Evaluate
    metrics = model.val(split='test')
    print("Fine-tuning complete. Metrics on rare test set:")
    print(metrics)

if __name__ == "__main__":
    fine_tune()