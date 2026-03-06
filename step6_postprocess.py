import hashlib, shutil, random, json
from pathlib import Path
from PIL import Image

BASE = Path.home() / "distraction_dataset"
IMG_SIZE = (224, 224)
CLASSES = ["focused", "distracted", "empty"]

# --- Resize all ---
print("Resizing all images to 224x224...")
for cls in CLASSES:
    cls_dir = BASE / cls
    for img_path in list(cls_dir.glob("*.*")):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(IMG_SIZE, Image.LANCZOS)
                out_path = img_path.with_suffix(".jpg")
                img.save(out_path, "JPEG", quality=90)
                if img_path.suffix.lower() != ".jpg":
                    img_path.unlink()
            except:
                img_path.unlink()

# --- Deduplicate ---
print("Removing duplicates...")
for cls in CLASSES:
    cls_dir = BASE / cls
    seen = set()
    removed = 0
    for img_path in sorted(cls_dir.glob("*.jpg")):
        h = hashlib.md5(img_path.read_bytes()).hexdigest()
        if h in seen:
            img_path.unlink()
            removed += 1
        else:
            seen.add(h)
    print(f"  {cls}: removed {removed} duplicates")

# --- Report ---
print("\nImage counts after dedup:")
for cls in CLASSES:
    n = len(list((BASE / cls).glob("*.jpg")))
    print(f"  {cls}: {n}")

# --- Train/Val/Test split (70/15/15) ---
print("\nCreating train/val/test splits...")
splits_dir = BASE / "splits"
random.seed(42)

for cls in CLASSES:
    for split in ["train", "val", "test"]:
        (splits_dir / split / cls).mkdir(parents=True, exist_ok=True)

    images = sorted((BASE / cls).glob("*.jpg"))
    random.shuffle(images)
    n = len(images)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    for img in images[:n_train]:
        shutil.copy2(img, splits_dir / "train" / cls / img.name)
    for img in images[n_train:n_train + n_val]:
        shutil.copy2(img, splits_dir / "val" / cls / img.name)
    for img in images[n_train + n_val:]:
        shutil.copy2(img, splits_dir / "test" / cls / img.name)

# --- Summary ---
print("\nFinal split counts:")
info = {"classes": CLASSES, "image_size": list(IMG_SIZE), "splits": {}}
for split in ["train", "val", "test"]:
    info["splits"][split] = {}
    for cls in CLASSES:
        n = len(list((splits_dir / split / cls).glob("*.jpg")))
        info["splits"][split][cls] = n
        print(f"  {split}/{cls}: {n}")

with open(BASE / "dataset_info.json", "w") as f:
    json.dump(info, f, indent=2)

print(f"\nDataset info saved to {BASE / 'dataset_info.json'}")
print(f"Dataset ready at: {splits_dir}")
