from ultralytics import YOLO
from pathlib import Path
import torch
import shutil
import sys
import multiprocessing  # Required for freeze_support()

# 1. Setup Project Paths (Outside main so workers can see them)
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_YAML_PATH = PROJECT_ROOT / "dataset" / "xview_stratified_dataset" / "xview_yolo.yaml"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def train_baseline():
    # 2. GPU Utilization
    device_to_use = 0 if torch.cuda.is_available() else 'cpu'
    if device_to_use == 0:
        print(f"DEBUG: Found GPU: {torch.cuda.get_device_name(0)}")

    # 3. Initialize Model
    try:
        model = YOLO("yolo26s.pt")
    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")
        sys.exit(1)

    # 4. Train: Scientific Baseline
    print(f"DEBUG: Starting training with data from: {DATA_YAML_PATH}")
    model.train(
        # --- CORE RUN SETTINGS ---
        data=str(DATA_YAML_PATH),      # Path to the xView dataset configuration file
        epochs=50,                     # 50 full passes; enough to see if mAP plateaus or climbs
        imgsz=640,                     # Standard resolution; higher is better for tiny xView objects
        batch=4,                       # 4 images per step; keeps VRAM usage stable on 6GB RTX 3060
        device=device_to_use,          # Uses '0' (GPU) for speed, falling back to 'cpu' if needed
        project=str(PROJECT_ROOT / "xView_Baseline"), # Main output directory
        name="baseline_run",           # Sub-directory for this specific experimental version
        seed=42,                       # Standard seed for reproducibility; ensures same initial weights
        deterministic=True,            # Forces use of deterministic algorithms (slower but reproducible)
        workers=4,                     # Number of CPU threads for data loading; 4 is safe for laptop RAM
        
        # --- GENERAL STRATEGY ---
        pretrained=True,               # Uses pretrained YOLOv26 weights; a "standard" industry baseline
        optimizer='auto',              # Framework picks the best optimizer (likely AdamW for this LR)
        patience=100,                  # Wait 100 epochs without improvement before stopping (set high)
        save=True,                     # Saves best.pt and last.pt checkpoints
        val=True,                      # Performs validation every epoch to monitor learning
        plots=True,                    # Generates charts (loss, mAP, confusion matrix) for your paper
        
        # --- SCIENTIFIC BASELINE: ALL TRICKS OFF ---
        # We set these to 0.0 or False to see "raw" data performance.
        augment=False,                 # Master switch: disables most online image-level augmentations
        auto_augment=None,             # Disables RandAugment policies (YOLO default is 'randaugment')
        erasing=0.0,                   # Disables Random Erasing (YOLO default is 0.4)
        mosaic=0.0,                    # Disables 4-image stitching; model sees only single xView chips
        mixup=0.0,                     # Disables blending two images; no synthetic "occlusions"
        copy_paste=0.0,                # Disables copying objects between images
        degrees=0.0,                   # No rotations; forces model to learn fixed-orientation features
        translate=0.0,                 # No image translation (random shifting)
        scale=0.0,                     # No random zooming; model sees objects at native satellite scale
        shear=0.0,                     # No image distortion/shearing
        perspective=0.0,               # No 3D-like perspective warping
        flipud=0.0,                    # No vertical flips (Up-Down)
        fliplr=0.0,                    # No horizontal flips (Left-Right)
        hsv_h=0.0,                     # No hue jitter; model learns fixed sensor color profiles
        hsv_s=0.0,                     # No saturation jitter
        hsv_v=0.0,                     # No brightness jitter; model sees fixed time-of-day lighting
        
        # --- LOSS PARAMETERS (Standard Weights) ---
        box=7.5,                       # Box loss gain; importance of being pixel-perfect on location
        cls=0.5,                       # Class loss gain; importance of getting the 60 classes right
        dfl=1.5,                       # Distribution Focal Loss; refines boxes for tiny, dense objects
    )

    # 5. Save Final Model
    best_weight = PROJECT_ROOT / "xView_Baseline" / "baseline_run" / "weights" / "best.pt"
    if best_weight.exists():
        shutil.copy2(best_weight, MODELS_DIR / "yolo26s_baseline.pt")
        print(f"SUCCESS: Baseline model saved to {MODELS_DIR / 'yolo26s_baseline.pt'}")

    # 6. Evaluation
    print("DEBUG: Final evaluation on Test split...")
    metrics = model.val(data=str(DATA_YAML_PATH), split="test")
    print(f"Test Result - mAP50: {metrics.box.map50:.4f}")

# --- THE CRITICAL WINDOWS PROTECTOR ---
if __name__ == '__main__':
    # This prevents the "bootstrapping" error on Windows
    multiprocessing.freeze_support()
    train_baseline()