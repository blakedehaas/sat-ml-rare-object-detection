#!/bin/bash
set -e

echo "Setting up yolo_venv"

# Rebuild the environment
python -m venv yolo_venv --clear
source yolo_venv/bin/activate

# Upgrade tooling
pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt

echo ""
echo "✅ Setup Complete!"
echo "To use this environment run:"
echo "source yolo_venv/bin/activate"
echo "----------------------------------------------------"