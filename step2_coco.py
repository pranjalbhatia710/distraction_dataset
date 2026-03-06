import fiftyone as fo
import fiftyone.zoo as foz
import shutil
from pathlib import Path

BASE = Path.home() / "distraction_dataset"
MAX = 1500

# --- FOCUSED: person + (book | laptop | keyboard), NO phone ---
print("Downloading COCO focused subset...")
ds = foz.load_zoo_dataset("coco-2017", split="validation", classes=["person","book","laptop","keyboard"], max_samples=MAX, dataset_name="coco_foc")
count = 0
for s in ds:
    labels = [d.label for d in s.ground_truth.detections]
    if "person" in labels and any(x in labels for x in ["book","laptop","keyboard"]) and "cell phone" not in labels:
        shutil.copy2(s.filepath, BASE/"focused"/f"coco_foc_{count:05d}.jpg")
        count += 1
print(f"Saved {count} focused images from COCO")
fo.delete_dataset("coco_foc")

# --- DISTRACTED: person + cell phone ---
print("Downloading COCO distracted subset...")
ds = foz.load_zoo_dataset("coco-2017", split="validation", classes=["person","cell phone"], max_samples=MAX, dataset_name="coco_dist")
count = 0
for s in ds:
    labels = [d.label for d in s.ground_truth.detections]
    if "person" in labels and "cell phone" in labels:
        shutil.copy2(s.filepath, BASE/"distracted"/f"coco_dist_{count:05d}.jpg")
        count += 1
print(f"Saved {count} distracted images from COCO")
fo.delete_dataset("coco_dist")

# --- EMPTY: furniture/desk scene, NO person ---
print("Downloading COCO empty subset...")
ds = foz.load_zoo_dataset("coco-2017", split="validation", classes=["chair","dining table","laptop"], max_samples=MAX, dataset_name="coco_emp")
count = 0
for s in ds:
    labels = [d.label for d in s.ground_truth.detections]
    if "person" not in labels:
        shutil.copy2(s.filepath, BASE/"empty"/f"coco_emp_{count:05d}.jpg")
        count += 1
print(f"Saved {count} empty images from COCO")
fo.delete_dataset("coco_emp")
