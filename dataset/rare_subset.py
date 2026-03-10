import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import yaml

class xViewHybridStratifier:
    def __init__(self, source_dir, output_dir, rare_indices, all_names, min_instances=500, seed=42):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.rare_indices = set(rare_indices)
        self.all_names = all_names
        self.min_instances = min_instances
        random.seed(seed)
        
        # metadata[split][filename] stores paths and class list
        self.metadata = {"train": {}, "val": {}, "test": {}}
        # Tracking which files are selected for the new dataset
        self.selected_files = {"train": set(), "val": set(), "test": set()}
        # Current object counts in our new subset
        self.current_counts = defaultdict(int)

    def load_metadata(self):
        """Crawls existing train, val, and test folders separately."""
        for split in ["train", "val", "test"]:
            img_folder = self.source_dir / "images" / split
            if not img_folder.exists():
                print(f"WARNING: Split folder {img_folder} not found. Skipping.")
                continue
                
            all_imgs = list(img_folder.glob("*.jpg"))
            for p in tqdm(all_imgs, desc=f"Scanning {split} source"):
                # Labels are in labels/split/filename.txt
                lbl_p = self.source_dir / "labels" / split / f"{p.stem}.txt"
                
                classes = []
                if lbl_p.exists():
                    with open(lbl_p, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                classes.append(int(parts[0]))
                
                self.metadata[split][p.name] = {
                    "img_p": p,
                    "lbl_p": lbl_p if lbl_p.exists() else None,
                    "classes": classes,
                    "has_rare": any(c in self.rare_indices for c in classes)
                }

    def _add_to_selection(self, fname, split):
        """Helper to safely add a file and update the global count for all its objects."""
        if fname not in self.selected_files[split]:
            self.selected_files[split].add(fname)
            for c in self.metadata[split][fname]["classes"]:
                self.current_counts[c] += 1

    def run_stratification(self):
        # --- PHASE 1: THE RARE GRAB ---
        print("\n[Phase 1] Extracting all images containing rare classes...")
        for split in ["train", "val", "test"]:
            for fname, meta in self.metadata[split].items():
                if meta["has_rare"]:
                    self._add_to_selection(fname, split)
        
        # --- PHASE 2: ITERATIVE COMMON CLASS BOLSTERING ---
        print(f"[Phase 2] Bolstering common classes to {self.min_instances} instances...")
        common_indices = set(self.all_names.keys()) - self.rare_indices
        
        while True:
            # Find common classes currently below the threshold
            under = {c: self.current_counts[c] for c in common_indices if self.current_counts[c] < self.min_instances}
            if not under:
                break # All common classes have at least min_instances representation
            
            # Pick the class that is currently most under-represented
            target_cls = min(under, key=under.get)
            needed = self.min_instances - self.current_counts[target_cls]
            
            # Pull extra images while respecting 80/10/10 split
            ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
            for split, ratio in ratios.items():
                split_needed = max(1, int(needed * ratio))
                
                # Candidates: Have the target class, NOT already selected, in this specific split
                candidates = [fn for fn, m in self.metadata[split].items() 
                             if target_cls in m["classes"] and fn not in self.selected_files[split]]
                
                random.shuffle(candidates)
                for fn in candidates[:split_needed]:
                    self._add_to_selection(fn, split)

    def generate_yaml(self):
        """Creates the xview_yolo.yaml with exact key order and relative paths."""
        output_yaml_path = self.output_dir / "xview_yolo.yaml"
        
        # We define the path as 'dataset/output_dir_name'
        # e.g., 'dataset/xview_rare_stratified_dataset'
        relative_path = f"dataset/{self.output_dir.name}"

        # Constructing the YAML as a string to maintain strict order
        yaml_content = f"path: {relative_path}\n"
        yaml_content += "train: images/train\n"
        yaml_content += "val: images/val\n"
        yaml_content += "test: images/test\n"
        yaml_content += f"nc: {len(self.all_names)}\n"
        yaml_content += "names:\n"
        
        # Sort names by index to ensure 0-59 order
        for i in sorted(self.all_names.keys()):
            yaml_content += f"  {i}: {self.all_names[i]}\n"

        with open(output_yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Generated formatted YAML at {output_yaml_path}")

    def report_and_transfer(self):
        print("\n" + "="*75)
        print(f"{'ID':<4} {'Class Name':<40} {'Count':<10} {'Type'}")
        print("-" * 75)
        for i in sorted(self.all_names.keys()):
            count = self.current_counts[i]
            ctype = "RARE" if i in self.rare_indices else "COMMON"
            name = self.all_names[i]
            print(f"{i:<4} {name:<40} {count:<10} {ctype}")
        print("="*75)

        # File Transfer
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        for split in ["train", "val", "test"]:
            for fname in tqdm(self.selected_files[split], desc=f"Transferring {split}"):
                m = self.metadata[split][fname]
                dest_img = self.output_dir / "images" / split / fname
                dest_lbl = self.output_dir / "labels" / split / f"{Path(fname).stem}.txt"
                
                dest_img.parent.mkdir(parents=True, exist_ok=True)
                dest_lbl.parent.mkdir(parents=True, exist_ok=True)
                
                shutil.copy2(m["img_p"], dest_img)
                if m["lbl_p"]:
                    shutil.copy2(m["lbl_p"], dest_lbl)
        
        self.generate_yaml()

def main():
    # --- CONFIGURATION ---
    RARE_CLASS_INDICES = [
        17, 36, 37, 3, 41, 45, 0, 59, 21, 32, 22, 54, 20, 15, 16, 34,
        33, 27, 29, 35, 26, 49, 44, 40, 1, 38, 58, 31, 30, 42,
        25, 46, 23, 28, 2, 12, 43, 14
    ]

    CLASS_NAMES = {
        0: 'Fixed-wing Aircraft', 1: 'Small Aircraft', 2: 'Passenger/Cargo Plane', 3: 'Helicopter',
        4: 'Passenger Vehicle', 5: 'Small Car', 6: 'Bus', 7: 'Pickup Truck', 8: 'Utility Truck',
        9: 'Truck', 10: 'Cargo Truck', 11: 'Truck Tractor w/ Box Trailer', 12: 'Truck Tractor',
        13: 'Trailer', 14: 'Truck Tractor w/ Flatbed Trailer', 15: 'Truck Tractor w/ Liquid Tank',
        16: 'Crane Truck', 17: 'Railway Vehicle', 18: 'Passenger Car', 19: 'Cargo/Container Car',
        20: 'Flat Car', 21: 'Tank car', 22: 'Locomotive', 23: 'Maritime Vessel', 24: 'Motorboat',
        25: 'Sailboat', 26: 'Tugboat', 27: 'Barge', 28: 'Fishing Vessel', 29: 'Ferry', 30: 'Yacht',
        31: 'Container Ship', 32: 'Oil Tanker', 33: 'Engineering Vehicle', 34: 'Tower crane',
        35: 'Container Crane', 36: 'Reach Stacker', 37: 'Straddle Carrier', 38: 'Mobile Crane',
        39: 'Dump Truck', 40: 'Haul Truck', 41: 'Scraper/Tractor', 42: 'Front loader/Bulldozer',
        43: 'Excavator', 44: 'Cement Mixer', 45: 'Ground Grader', 46: 'Hut/Tent', 47: 'Shed',
        48: 'Building', 49: 'Aircraft Hangar', 50: 'Damaged Building', 51: 'Facility',
        52: 'Construction Site', 53: 'Vehicle Lot', 54: 'Helipad', 55: 'Storage Tank',
        56: 'Shipping container lot', 57: 'Shipping Container', 58: 'Pylon', 59: 'Tower'
    }

    stratifier = xViewHybridStratifier(
        source_dir="xview_stratified_dataset",
        output_dir="xview_rare_stratified_dataset",
        rare_indices=RARE_CLASS_INDICES,
        all_names=CLASS_NAMES,
        min_instances=1  # Bolster common classes to this floor
    )
    
    stratifier.load_metadata()
    stratifier.run_stratification()
    stratifier.report_and_transfer()
    print("\nSUCCESS: xView rare stratified dataset created.")

if __name__ == "__main__":
    main()