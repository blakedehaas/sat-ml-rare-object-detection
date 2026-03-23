import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

class xViewStratifier:
    def __init__(self, source_dir, output_dir, seed=42):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        random.seed(seed)
        
        self.file_metadata = {}
        self.assignments = {} 
        self.global_counts = {"train": 0, "val": 0, "test": 0}

        print(f"DEBUG: Initializing Stratifier")
        print(f"DEBUG: Source Directory: {self.source_dir.resolve()}")

    def load_metadata(self):
        print("DEBUG: Crawling directory for images...")
        # Since images and labels are mixed, we target the image folder directly
        image_search_path = self.source_dir / "images"
        all_image_paths = list(image_search_path.glob("*.jpg"))
        
        print(f"DEBUG: Found {len(all_image_paths)} images. Parsing labels from same directory...")

        for img_path in tqdm(all_image_paths, desc="Parsing Metadata"):
            filename = img_path.name
            # Labels are in the same folder as images according to your structure
            label_path = img_path.with_suffix('.txt')
            
            classes = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            classes.append(int(parts[0]))
            
            self.file_metadata[filename] = {
                "img_path": img_path,
                "lbl_path": label_path if label_path.exists() else None,
                "classes": list(set(classes)) 
            }

    def _get_current_class_distribution(self, unassigned_images):
        counts = defaultdict(list)
        for fname in unassigned_images:
            for cls in self.file_metadata[fname]["classes"]:
                counts[cls].append(fname)
        return counts

    def run_stratification(self):
        unassigned_images = set(self.file_metadata.keys())
        if not unassigned_images:
            print("ERROR: No images found to process. Check your source path.")
            return

        total_initial = len(unassigned_images)
        test_cap = int(total_initial * 0.10)
        val_cap = int(total_initial * 0.10)

        print(f"DEBUG: Starting Iterative Loop. Targets: Test={test_cap}, Val={val_cap}")

        while True:
            current_dist = self._get_current_class_distribution(unassigned_images)
            if not current_dist:
                break 
            
            # Identify rarest remaining class
            rarest_cls = min(current_dist.keys(), key=lambda c: len(current_dist[c]))
            images_to_process = current_dist[rarest_cls]
            random.shuffle(images_to_process)
            
            num_available = len(images_to_process)
            cls_test_target = max(1, int(num_available * 0.10))
            cls_val_target = max(1, int(num_available * 0.10))
            
            cls_assigned = {"train": 0, "val": 0, "test": 0}

            for img in images_to_process:
                if cls_assigned["test"] < cls_test_target and self.global_counts["test"] < test_cap:
                    target = "test"
                elif cls_assigned["val"] < cls_val_target and self.global_counts["val"] < val_cap:
                    target = "val"
                else:
                    target = "train"

                self.assignments[img] = target
                self.global_counts[target] += 1
                cls_assigned[target] += 1
                unassigned_images.remove(img)

        # Assign background images (no labels)
        remaining_background = list(unassigned_images)
        print(f"DEBUG: Assigning {len(remaining_background)} background images...")
        random.shuffle(remaining_background)
        for img in remaining_background:
            if self.global_counts["test"] < test_cap:
                target = "test"
            elif self.global_counts["val"] < val_cap:
                target = "val"
            else:
                target = "train"
            self.assignments[img] = target
            self.global_counts[target] += 1

        print(f"DEBUG: Final Global Totals -> {self.global_counts}")

    def execute_file_transfer(self):
        print(f"DEBUG: Moving files to YOLO structure at {self.output_dir}...")
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        for filename, split in tqdm(self.assignments.items(), desc="Transferring"):
            # YOLO standard: images and labels in separate top-level folders
            dest_img_path = self.output_dir / "images" / split / filename
            dest_lbl_path = self.output_dir / "labels" / split / f"{Path(filename).stem}.txt"
            
            dest_img_path.parent.mkdir(parents=True, exist_ok=True)
            dest_lbl_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy Image
            shutil.copy2(self.file_metadata[filename]["img_path"], dest_img_path)
            
            # Copy Label
            src_lbl = self.file_metadata[filename]["lbl_path"]
            if src_lbl:
                shutil.copy2(src_lbl, dest_lbl_path)

def main():
    stratifier = xViewStratifier(
        source_dir="xview_cleaned_dataset",
        output_dir="xview_stratified_dataset"
    )
    stratifier.load_metadata()
    stratifier.run_stratification()
    stratifier.execute_file_transfer()
    print("FINISH: Stratification complete.")

if __name__ == "__main__":
    main()