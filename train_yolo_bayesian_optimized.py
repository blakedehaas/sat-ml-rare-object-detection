import os
import wandb
import torch
from pathlib import Path
from ultralytics import YOLO

# ---------- CONFIGURATION ----------
ENTITY = "blakedehaas-auroral-precipitation-ml"
PROJECT = "xview-bayesian-optimized-training"
DATASET_ROOT = Path("scratch/alpine/blde8334/sat-ml-rare-object-detection/datasets/xview_stratified_dataset")
DATA_YAML = DATASET_ROOT / "xview_yolo.yaml"
MODEL_BASE = "yolo26s.pt"

BEST_CONFIG = {
    'epochs': 100,
    'patience': 25,
    'amp': False,           
    'imgsz': 640,
    'batch': 64,
    'lr0': 0.000634891670663402 # best sweep,
    'box': 13.45884042445332,
    'cls': 1.42040856867997,
    'mosaic': 0.9215140956772788,
    'close_mosaic': 10,     
    'seed': 42,
    'deterministic': True,      
    'workers': 8,
    'exist_ok': True
    }

def main():
    # 1. Initialize W&B
    wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name="xview_train_bayesian_optimized",
        job_type="bayesian_optimized_training",
        config=BEST_CONFIG
    )

    # 2. Load Model
    model = YOLO(MODEL_BASE)
    
    print("\n" + "="*60)
    print(f" LAUNCHING BAYESIAN OPTIMIZED TRAINING RUN")
    print("="*60 + "\n")

    # 4. Training Call
    try:
        model.train(
            data=str(DATA_YAML),
            project=PROJECT,
            name="xview_train_bayesian_optimized",
            **BEST_CONFIG
        )
    finally:
        if wandb.run:
            wandb.finish()

if __name__ == "__main__":
    main()