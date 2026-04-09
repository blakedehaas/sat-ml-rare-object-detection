#!/bin/bash
#SBATCH --job-name=fine_tune_yolo
#SBATCH --partition=atesting_mi100
#SBATCH --qos=testing
#SBATCH --gres=gpu:mi100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=fine_tune_yolo.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brne5584@colorado.edu
# -------------------------
# Move to project directory
# -------------------------
cd /scratch/alpine/brne5584/sat-ml-rare-object-detection || exit 1
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
# Run fine-tuning
# -------------------------
echo "Starting fine-tuning at: $(date)"
stdbuf -oL -eL python yolo_fine_tune.py
echo "Job finished at: $(date)"