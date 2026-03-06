import fiftyone as fo
import fiftyone.zoo as foz
import shutil
from pathlib import Path

BASE = Path.home() / "distraction_dataset"
MAX = 1500

# Person + Mobile phone -> distracted
print("Downloading Open Images phone subset...")
try:
    ds = foz.load_zoo_dataset("open-images-v7", split="validation", classes=["Mobile phone","Person"], max_samples=MAX, dataset_name="oi_phone")
    count = 0
    for s in ds:
        if s.ground_truth and s.ground_truth.detections:
            labels = [d.label for d in s.ground_truth.detections]
            if "Mobile phone" in labels and "Person" in labels:
                shutil.copy2(s.filepath, BASE/"distracted"/f"oi_dist_{count:05d}.jpg")
                count += 1
    print(f"Saved {count} distracted from Open Images")
    fo.delete_dataset("oi_phone")
except Exception as e:
    print(f"Open Images phone failed: {e}")

# Person + Book/Laptop -> focused
print("Downloading Open Images focused subset...")
try:
    ds = foz.load_zoo_dataset("open-images-v7", split="validation", classes=["Book","Laptop","Person"], max_samples=MAX, dataset_name="oi_foc")
    count = 0
    for s in ds:
        if s.ground_truth and s.ground_truth.detections:
            labels = [d.label for d in s.ground_truth.detections]
            if "Person" in labels and ("Book" in labels or "Laptop" in labels) and "Mobile phone" not in labels:
                shutil.copy2(s.filepath, BASE/"focused"/f"oi_foc_{count:05d}.jpg")
                count += 1
    print(f"Saved {count} focused from Open Images")
    fo.delete_dataset("oi_foc")
except Exception as e:
    print(f"Open Images focused failed: {e}")
