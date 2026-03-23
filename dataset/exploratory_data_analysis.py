import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
from pathlib import Path
from collections import Counter
import numpy as np

# ================= CONFIGURATION =================
DATA_ROOT = Path("xview_stratified_dataset")
LABELS_DIR = DATA_ROOT / "labels"
IMAGES_DIR = DATA_ROOT / "images"
OUTPUT_DIR = Path("eda_stratified_output")
OUTPUT_MD = OUTPUT_DIR / "stratified_exploratory_data_analysis.md"

# --- ANALYSIS FILTER ---
# Add IDs here to filter specific classes (e.g., [0, 4]), or [] for ALL.
TARGET_CLASSES = [] 

CLASS_NAMES = {
    0: "Fixed-wing Aircraft", 1: "Small Aircraft", 2: "Passenger/Cargo Plane", 3: "Helicopter",
    4: "Passenger Vehicle", 5: "Small Car", 6: "Bus", 7: "Pickup Truck", 8: "Utility Truck",
    9: "Truck", 10: "Cargo Truck", 11: "Truck Tractor w/ Box Trailer", 12: "Truck Tractor",
    13: "Trailer", 14: "Truck Tractor w/ Flatbed Trailer", 15: "Truck Tractor w/ Liquid Tank",
    16: "Crane Truck", 17: "Railway Vehicle", 18: "Passenger Car", 19: "Cargo/Container Car",
    20: "Flat Car", 21: "Tank car", 22: "Locomotive", 23: "Maritime Vessel", 24: "Motorboat",
    25: "Sailboat", 26: "Tugboat", 27: "Barge", 28: "Fishing Vessel", 29: "Ferry", 30: "Yacht",
    31: "Container Ship", 32: "Oil Tanker", 33: "Engineering Vehicle", 34: "Tower crane",
    35: "Container Crane", 36: "Reach Stacker", 37: "Straddle Carrier", 38: "Mobile Crane",
    39: "Dump Truck", 40: "Haul Truck", 41: "Scraper/Tractor", 42: "Front loader/Bulldozer",
    43: "Excavator", 44: "Cement Mixer", 45: "Ground Grader", 46: "Hut/Tent", 47: "Shed",
    48: "Building", 49: "Aircraft Hangar", 50: "Damaged Building", 51: "Facility",
    52: "Construction Site", 53: "Vehicle Lot", 54: "Helipad", 55: "Storage Tank",
    56: "Shipping container lot", 57: "Shipping Container", 58: "Pylon", 59: "Tower"
}

# =================================================

def ensure_dirs():
    if not OUTPUT_DIR.exists():
        os.makedirs(OUTPUT_DIR)

def get_file_hash(file_path):
    """Generates an MD5 hash for a file to check for identical content."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        return f"ERROR_{e}"

def perform_sanity_check():
    """Checks for data leakage using Filenames and MD5 Hashes."""
    print("Performing deep content sanity check (Hashing)...")
    splits = ["train", "val", "test"]
    fingerprints = {}
    leakage_found = []

    for split in splits:
        split_path = IMAGES_DIR / split
        if not split_path.exists(): continue
            
        for img_file in split_path.iterdir():
            if img_file.is_file():
                f_hash = get_file_hash(img_file)
                if f_hash in fingerprints:
                    orig_name, orig_split = fingerprints[f_hash]
                    leakage_found.append(f"Leak: `{img_file.name}` ({split}) identical to `{orig_name}` ({orig_split})")
                else:
                    fingerprints[f_hash] = (img_file.name, split)

    if not leakage_found:
        return ["**Status: CLEAN**", "✅ **PASS**: No identical file content detected via MD5 hashing."]
    return ["**Status: CRITICAL WARNING**", f"❌ **FAIL**: Found {len(leakage_found)} identical image contents."] + leakage_found

def parse_labels():
    print(f"Parsing labels from {DATA_ROOT}...")
    data = []
    for split in ["train", "val", "test"]:
        split_path = LABELS_DIR / split
        if not split_path.exists(): continue
        files = list(split_path.glob("*.txt"))
        for file in files:
            with open(file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5: continue
                    cls_id = int(parts[0])
                    if TARGET_CLASSES and cls_id not in TARGET_CLASSES: continue
                    data.append({
                        "split": split, 
                        "file": file.name,
                        "class_id": cls_id, 
                        "class_name": CLASS_NAMES.get(cls_id, f"ID-{cls_id}"),
                        "x": float(parts[1]), "y": float(parts[2]), "w": float(parts[3]), "h": float(parts[4])
                    })
    return pd.DataFrame(data)

def generate_visualizations(df):
    print("Generating split-specific plots...")
    sns.set_style("whitegrid")
    
    # 1. Combined Class Distribution (Logistic X-Axis)
    plt.figure(figsize=(14, 12))
    order = df['class_name'].value_counts().index
    ax = sns.countplot(data=df, y='class_name', hue='split', order=order, hue_order=['train', 'val', 'test'], palette='viridis')
    ax.set_xscale("log")
    plt.title("Class Instance Distribution Across Splits (Log Scale)")
    plt.xlabel("Instance Count (Log10)")
    plt.ylabel("Class Name")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution_log.png")
    plt.close()

    # 2. Separate Scatter & Heatmaps per Split
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        if split_df.empty: continue

        # BBox Dimensions
        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=split_df, x='w', y='h', alpha=0.3, s=15)
        plt.xlim(0, 1); plt.ylim(0, 1)
        plt.title(f"BBox Dimensions (W vs H) - {split.upper()}")
        plt.savefig(OUTPUT_DIR / f"bbox_{split}.png")
        plt.close()

        # Spatial Heatmap
        plt.figure(figsize=(6, 6))
        plt.hist2d(split_df['x'], split_df['y'], bins=50, range=[[0, 1], [0, 1]], cmap='magma')
        plt.title(f"Spatial Object Density - {split.upper()}")
        plt.savefig(OUTPUT_DIR / f"heatmap_{split}.png")
        plt.close()

def create_markdown_report(df, sanity_results):
    print("Finalizing Markdown report...")
    
    # Instance Counts (Total objects)
    instance_pivot = df.groupby(['class_name', 'split']).size().unstack(fill_value=0)
    
    # Image Counts (Unique images containing the class)
    image_pivot = df.groupby(['class_name', 'split'])['file'].nunique().unstack(fill_value=0)
    
    # Combine into a master report table
    report_df = pd.DataFrame(index=instance_pivot.index)
    for split in ['train', 'val', 'test']:
        if split in instance_pivot.columns:
            report_df[f'{split.capitalize()} (Inst)'] = instance_pivot[split]
            report_df[f'{split.capitalize()} (Imgs)'] = image_pivot[split]
        else:
            report_df[f'{split.capitalize()} (Inst)'] = 0
            report_df[f'{split.capitalize()} (Imgs)'] = 0
            
    report_df['Total Instances'] = instance_pivot.sum(axis=1)
    report_df = report_df.sort_values(by='Total Instances', ascending=False)

    sanity_md = "\n".join([f"* {line}" for line in sanity_results])

    md_content = f"""# xView EDA Report: Advanced Stratification & Leakage Check
**Dataset Root:** `{DATA_ROOT}`  
**Class Filter:** {"All 60 Classes" if not TARGET_CLASSES else f"Filtered to IDs: {TARGET_CLASSES}"}

## 1. Deep Sanity Check (MD5 Content Hashing)
{sanity_md}

---

## 2. Iterative Stratification Summary
This table compares **Instances** (total boxes) vs **Images** (unique files). 
*Classes with high (Inst) but low (Imgs) are "clustered" and harder to split perfectly.*

{report_df.to_markdown()}

---

## 3. Class Distribution (Logarithmic Scale)
![Class Distribution Log](class_distribution_log.png)

---

## 4. Split Comparison (Visual Integrity)

| Split | BBox Dimensions (Normalized) | Spatial Object Heatmap |
| :--- | :--- | :--- |
| **TRAIN** | ![Train BBox](bbox_train.png) | ![Train Heatmap](heatmap_train.png) |
| **VAL** | ![Val BBox](bbox_val.png) | ![Val Heatmap](heatmap_val.png) |
| **TEST** | ![Test BBox](bbox_test.png) | ![Test Heatmap](heatmap_test.png) |

---
*Generated by Geospatial EDA Suite v2026.1*
"""
    # Force UTF-8 encoding to prevent Windows cp1252 errors with emojis
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)

def main():
    ensure_dirs()
    sanity = perform_sanity_check()
    df = parse_labels()
    
    if df.empty:
        return print("No data found! Check your directory structure or class filter.")
        
    generate_visualizations(df)
    create_markdown_report(df, sanity)
    print(f"\nAnalysis Successful.")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()