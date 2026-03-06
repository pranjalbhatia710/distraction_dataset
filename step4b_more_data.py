import fiftyone as fo
import fiftyone.zoo as foz
import shutil
from pathlib import Path

BASE = Path.home() / "distraction_dataset"

# --- Open Images: try with correct field names ---
print("Open Images — trying different field names...")
try:
    ds = foz.load_zoo_dataset("open-images-v7", split="validation",
                              classes=["Mobile phone", "Person"],
                              max_samples=1500, dataset_name="oi_phone2")
    count = 0
    for s in ds:
        # Try multiple possible field names
        dets = None
        for field in ["ground_truth", "detections", "positive_labels", "classifications"]:
            if s.has_field(field):
                val = s[field]
                if hasattr(val, "detections"):
                    dets = val.detections
                elif hasattr(val, "classifications"):
                    dets = val.classifications
                break
        if dets:
            labels = [d.label for d in dets]
            if "Mobile phone" in labels and "Person" in labels:
                shutil.copy2(s.filepath, BASE / "distracted" / f"oi2_dist_{count:05d}.jpg")
                count += 1
    print(f"Saved {count} distracted from Open Images (phone)")
    fo.delete_dataset("oi_phone2")
except Exception as e:
    print(f"Open Images phone attempt 2 failed: {e}")

# --- More COCO focused from TRAIN split ---
print("\nDownloading more focused images from COCO train split...")
try:
    ds = foz.load_zoo_dataset("coco-2017", split="train",
                              classes=["person", "book", "laptop", "keyboard"],
                              max_samples=3000, dataset_name="coco_foc_train")
    count = 0
    for s in ds:
        labels = [d.label for d in s.ground_truth.detections]
        if "person" in labels and any(x in labels for x in ["book", "laptop", "keyboard"]) and "cell phone" not in labels:
            shutil.copy2(s.filepath, BASE / "focused" / f"coco_foc_t_{count:05d}.jpg")
            count += 1
            if count >= 400:
                break
    print(f"Saved {count} focused from COCO train")
    fo.delete_dataset("coco_foc_train")
except Exception as e:
    print(f"COCO train focused failed: {e}")

# --- More COCO distracted from TRAIN split ---
print("Downloading more distracted from COCO train split...")
try:
    ds = foz.load_zoo_dataset("coco-2017", split="train",
                              classes=["person", "cell phone"],
                              max_samples=3000, dataset_name="coco_dist_train")
    count = 0
    for s in ds:
        labels = [d.label for d in s.ground_truth.detections]
        if "person" in labels and "cell phone" in labels:
            shutil.copy2(s.filepath, BASE / "distracted" / f"coco_dist_t_{count:05d}.jpg")
            count += 1
            if count >= 300:
                break
    print(f"Saved {count} distracted from COCO train")
    fo.delete_dataset("coco_dist_train")
except Exception as e:
    print(f"COCO train distracted failed: {e}")

# Final count
print("\nUpdated counts:")
for cls in ["focused", "distracted", "empty"]:
    n = len(list((BASE / cls).glob("*.jpg")))
    print(f"  {cls}: {n}")
