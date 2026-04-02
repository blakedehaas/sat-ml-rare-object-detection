#!/bin/bash

#SBATCH --job-name=train_yolo_bayesian_optimized
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=train_yolo_bayesian_optimized.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=blake.dehaas@colorado.edu

# -------------------------
# Move to project directory
# -------------------------
cd /scratch/alpine/blde8334/sat-ml-rare-object-detection/sat-ml-rare-object-detection || exit 1

# -------------------------
# Module setup (REQUIRED on Alpine)
# -------------------------
module purge
module load gcc
module load cuda
module load python/3.10

# -------------------------
# Activate virtual environment
# -------------------------
source yolo_venv/bin/activate

# -------------------------
# Sanity checks (recommended)
# -------------------------
echo "Job started on $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

python - <<EOF
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

# -------------------------
# Run training
# -------------------------
echo "Starting training at: $(date)"
stdbuf -oL -eL python train_yolo_bayesian_optimized.py
echo "Job finished at: $(date)"