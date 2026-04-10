import os
import shutil
import torch
import yaml
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent
RARE_DATASET_YAML = PROJECT_ROOT / "dataset/xview_rare_stratified_dataset/xview_yolo.yaml"
BASE_MODEL_PATH = PROJECT_ROOT / "best_baseline.pt"
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
    """Find the latest epoch checkpoint across all fine-tune run folders."""
    base_dir = PROJECT_ROOT / "xview_rare"

    if not base_dir.exists():
        return None, 0

    # Search all fine_tune_weighted_from_epochN folders
    all_checkpoints = list(base_dir.glob("fine_tune_weighted_from_epoch*/weights/epoch*.pt"))

    if not all_checkpoints:
        return None, 0

    # Find the checkpoint with the highest epoch number
    latest = max(all_checkpoints, key=lambda p: int(p.stem.replace("epoch", "")))
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

    # 3. Fine-tune into a new folder for this run
    run_name = f"fine_tune_weighted_from_epoch{start_epoch}"
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
        name=run_name,
        exist_ok=True,
        save_period=1,
    )

    # 4. Rename checkpoints to reflect true epoch numbers
    weights_dir = PROJECT_ROOT / f"xview_rare/{run_name}/weights"
    local_checkpoints = sorted(
        weights_dir.glob("epoch*.pt"),
        key=lambda p: int(p.stem.replace("epoch", ""))
    )

    for local_ckpt in local_checkpoints:
        local_epoch = int(local_ckpt.stem.replace("epoch", ""))
        true_epoch = start_epoch + local_epoch + 1
        new_name = weights_dir / f"epoch{true_epoch}.pt"
        print(f"Renaming {local_ckpt.name} -> {new_name.name}")
        shutil.move(str(local_ckpt), str(new_name))

    # 5. Evaluate
    metrics = model.val(split='test')
    print("Fine-tuning complete. Metrics on rare test set:")
    print(metrics)

if __name__ == "__main__":
    fine_tune()