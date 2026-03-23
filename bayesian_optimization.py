import os
import wandb
from pathlib import Path
from ultralytics import YOLO

# ---------- GLOBAL VARIABLES ----------
DATASET_NAME = "xview_stratified_dataset"
MODEL_FILE = "yolo11s.pt" # Updated to yolo11s for the 2024/2025 standard, or keep yolo26s if specific to your build
PROJECT_NAME = "xview-bayesian-optimization"

# 1. Define the Sweep Configuration
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'metrics/mAP50(B)', 'goal': 'maximize'},
    'parameters': {
        # Fixed Params (Optional: move these here so they are logged)
        'imgsz': {'value': 640},
        'batch': {'value': 16},
        'epochs': {'value': 5}, # Keep sweep epochs low to save time
        
        # Training Hyperparameters
        'lr0': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1},
        'lrf': {'distribution': 'uniform', 'min': 0.01, 'max': 1.0},
        'momentum': {'distribution': 'uniform', 'min': 0.6, 'max': 0.98},
        'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-3},

        # Loss Gains
        'box': {'distribution': 'uniform', 'min': 1.0, 'max': 20.0},
        'cls': {'distribution': 'uniform', 'min': 0.2, 'max': 4.0},
        'dfl': {'distribution': 'uniform', 'min': 0.5, 'max': 3.0},

        # Augmentation
        'hsv_h': {'distribution': 'uniform', 'min': 0.0, 'max': 0.1},
        'hsv_s': {'distribution': 'uniform', 'min': 0.0, 'max': 0.9},
        'hsv_v': {'distribution': 'uniform', 'min': 0.0, 'max': 0.9},
        'degrees': {'distribution': 'uniform', 'min': 0.0, 'max': 180.0},
        'translate': {'distribution': 'uniform', 'min': 0.0, 'max': 0.2},
        'scale': {'distribution': 'uniform', 'min': 0.0, 'max': 0.9},
        'flipud': {'distribution': 'uniform', 'min': 0.0, 'max': 0.5},
        'fliplr': {'distribution': 'uniform', 'min': 0.0, 'max': 0.5},
        'mosaic': {'distribution': 'uniform', 'min': 0.0, 'max': 1.0},
        'mixup': {'distribution': 'uniform', 'min': 0.0, 'max': 0.3},
        'copy_paste': {'distribution': 'uniform', 'min': 0.0, 'max': 0.3}
    }
}

# ---------- SETUP ENVIRONMENT ----------
DATASET_ROOT = Path(f"dataset/{DATASET_NAME}").resolve()
DATA_YAML = DATASET_ROOT / "xview_yolo.yaml"

if not DATA_YAML.exists():
    raise FileNotFoundError(f"YAML not found at {DATA_YAML}")

# ---------- SWEEP WORKER FUNCTION ----------
def train_worker():
    with wandb.init() as run:
        # Use a fresh model instance for every sweep run to prevent weight leakage
        model = YOLO(MODEL_FILE)
        
        # .train() accepts a dict of hyperparams. 
        # We pass everything from wandb.config directly.
        results = model.train(
            data=str(DATA_YAML),
            project=PROJECT_NAME,
            name=f"sweep_run_{run.id}",
            **dict(wandb.config) # Unpacks all hyperparameters automatically
        )

# ---------- MAIN EXECUTION ----------
if __name__ == "__main__":
    wandb.login()

    # 2. Initialize and Run the Sweep
    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
    
    # Run 10 trials (adjust count as needed for your time/budget)
    wandb.agent(sweep_id, function=train_worker, count=10)

    # 3. Retrieve Optimal Hyperparameters
    print("\nSweep complete. Fetching best hyperparameters...")
    api = wandb.Api()
    sweep = api.sweep(f"{wandb.run.entity}/{PROJECT_NAME}/{sweep_id}")
    best_run = sweep.best_run()
    best_config = best_run.config

    # Remove W&B internal keys that YOLO won't recognize
    clean_best_config = {k: v for k, v in best_config.items() if not k.startswith('_')}
    
    print(f"✓ Best mAP50 found: {best_run.summary.get('metrics/mAP50(B)')}")
    print(f"✓ Best Config: {clean_best_config}")

    # 4. Final Training with Optimal Settings
    print("\nStarting final training with optimal hyperparameters...")
    final_model = YOLO(MODEL_FILE)
    
    # Override epochs for the final "golden" model
    clean_best_config['epochs'] = 50 
    
    final_model.train(
        data=str(DATA_YAML),
        project=PROJECT_NAME,
        name="final_optimal_model",
        **clean_best_config
    )