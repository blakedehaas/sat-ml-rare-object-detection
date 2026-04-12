import json
import yaml
import torch
from ultralytics import YOLO
from pathlib import Path
import numpy as np

RARE_CLASS_INDICES = set([
    17, 36, 37, 3, 41, 45, 0, 59, 21, 32, 22, 54, 20, 15, 16, 34,
    33, 27, 29, 35, 26, 49, 44, 40, 1, 38, 58, 31, 30, 42,
    25, 46, 23, 28, 2, 12, 43, 14
])

def load_config(config_path):
    """Load config.json containing model and dataset paths."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def calculate_mean_per_class_map50(metrics, rare_indices):
    """
    Calculate mean per-class mAP50.
    For each class, compute AP at IoU=0.50, then take the mean across all 60 classes.
    """
    ap50_per_class = [metrics.box.ap50[i] for i in range(len(metrics.names))]
    print(f"DEBUG box.ap50 length: {len(ap50_per_class)}")
    print(f"DEBUG box.ap50 values: {ap50_per_class}")
    
    maps_per_class = list(metrics.maps)
    print(f"DEBUG maps sample: {maps_per_class[:5]}")
    print(f"DEBUG maps length: {len(maps_per_class)}")

    rare_aps = [ap50_per_class[i] for i in range(len(metrics.names))
                if i in rare_indices]
    common_aps = [ap50_per_class[i] for i in range(len(metrics.names))
                  if i not in rare_indices]

    rare_mean = np.mean(rare_aps) if rare_aps else 0.0
    common_mean = np.mean(common_aps) if common_aps else 0.0

    # Mean per-class mAP50 = sum of all per-class AP50 / total number of classes
    all_aps = ap50_per_class
    mean_per_class_map50 = np.mean(all_aps)

    return mean_per_class_map50, ap50_per_class, rare_mean, common_mean

def evaluate_model(model_path, dataset_yaml, split='val'):
    """Load model and run evaluation on the given dataset split."""
    print(f"\nLoading model: {model_path}")
    print(f"Dataset: {dataset_yaml}")
    print(f"Split: {split}")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(dataset_yaml),
        split=split,
        verbose=False
    )
    return metrics

def format_results(metrics, dataset_name, model_name):
    """Format and return results as a string."""
    map50 = metrics.results_dict['metrics/mAP50(B)']
    weighted_mean, ap50_per_class, rare_mean, common_mean = calculate_mean_per_class_map50(
        metrics, RARE_CLASS_INDICES
    )

    lines = []
    lines.append("=" * 70)
    lines.append(f"Model:   {model_name}")
    lines.append(f"Dataset: {dataset_name}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("  --- Overall Metrics ---")
    lines.append(f"  mAP@50:                        {map50:.4f} ({map50*100:.2f}%)")
    lines.append(f"  Mean Per-Class mAP@50:         {weighted_mean:.4f} ({weighted_mean*100:.2f}%)")
    lines.append(f"    Common Class Mean mAP@50:    {common_mean:.4f} ({common_mean*100:.2f}%)")
    lines.append(f"    Rare Class Mean mAP@50:      {rare_mean:.4f} ({rare_mean*100:.2f}%)")
    lines.append("")
    lines.append("  --- Per-Class Results ---")
    lines.append(f"  {'ID':<4} {'Class Name':<40} {'mAP@50':<10} {'Instances':<12} {'Type'}")
    lines.append("  " + "-" * 70)

    for i, name in metrics.names.items():
        ap = ap50_per_class[i] if i < len(ap50_per_class) else 0.0
        instances = metrics.nt_per_class[i] if i < len(metrics.nt_per_class) else 0
        ctype = "RARE" if i in RARE_CLASS_INDICES else "COMMON"
        lines.append(f"  {i:<4} {name:<40} {ap:.4f}     {instances:<12} {ctype}")

    lines.append("=" * 70)
    return "\n".join(lines)

def run_evaluation(config_path, output_path):
    """Main evaluation function."""
    config = load_config(config_path)

    model_path = Path(config['model_path'])
    dataset_yaml = Path(config['dataset_yaml'])
    model_name = config.get('model_name', model_path.stem)
    dataset_name = config.get('dataset_name', dataset_yaml.parent.name)

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return
    if not dataset_yaml.exists():
        print(f"ERROR: Dataset YAML not found at {dataset_yaml}")
        return

    # Run evaluation
    metrics = evaluate_model(model_path, dataset_yaml, split='val')

    # Format results
    results_str = format_results(metrics, dataset_name, model_name)
    print(results_str)

    # Save to output file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(results_str)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <config.json> <output.txt>")
        sys.exit(1)

    config_path = sys.argv[1]
    output_path = sys.argv[2]
    run_evaluation(config_path, output_path)