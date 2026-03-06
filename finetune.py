#!/usr/bin/env python3
"""
Fine-tune the distraction detector for higher accuracy.

Improvements over initial training:
  - Starts from best_distraction_model.pth (warm start)
  - Lower learning rate with warmup
  - Stronger augmentation (RandAugment, erasing)
  - Class-weighted loss to handle imbalance
  - Longer training with cosine annealing
  - Label smoothing
  - Gradient clipping

Usage:
  python finetune.py --epochs 40
  python finetune.py --epochs 40 --lr 3e-4
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from pathlib import Path
import numpy as np
from collections import Counter

def get_class_weights(dataset):
    """Compute inverse-frequency weights for balanced sampling."""
    targets = [s[1] for s in dataset.samples]
    counts = Counter(targets)
    total = len(targets)
    weights_per_class = {c: total / (len(counts) * n) for c, n in counts.items()}
    sample_weights = [weights_per_class[t] for t in targets]
    return sample_weights, weights_per_class

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data)

    # Stronger augmentation for fine-tuning
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.05),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(data_dir / "train", train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", val_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", val_tf)

    print(f"Classes: {train_ds.classes}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Balanced sampling
    sample_weights, class_weights = get_class_weights(train_ds)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    print(f"Class weights: { {train_ds.classes[k]: f'{v:.2f}' for k, v in class_weights.items()} }")

    train_dl = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=32, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=32, num_workers=4, pin_memory=True)

    # Load existing model
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 3)

    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print(f"WARNING: {ckpt} not found — training from ImageNet weights")
        model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 3)

    model = model.to(device)

    # Label smoothing cross-entropy
    loss_weights = torch.tensor([class_weights[i] for i in range(3)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=0.1)

    # Differential learning rates: backbone lower, classifier higher
    backbone_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if "classifier" in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},      # backbone: 10x lower
        {"params": classifier_params, "lr": args.lr},           # classifier: full lr
    ], weight_decay=0.02)

    # Cosine annealing with warmup
    warmup_epochs = min(5, args.epochs // 4)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = 0
    patience_counter = 0
    patience = 12

    print(f"\nStarting fine-tuning for {args.epochs} epochs")
    print(f"  LR: backbone={args.lr * 0.1:.1e}, classifier={args.lr:.1e}")
    print(f"  Warmup: {warmup_epochs} epochs")
    print(f"  Patience: {patience}\n")

    for epoch in range(args.epochs):
        # Train
        model.train()
        loss_sum, correct, total = 0, 0, 0
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += loss.item()
            correct += (out.argmax(1) == lbls).sum().item()
            total += lbls.size(0)

        # Validate
        model.eval()
        vc, vt = 0, 0
        val_loss = 0
        with torch.no_grad():
            for imgs, lbls in val_dl:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                val_loss += nn.CrossEntropyLoss()(out, lbls).item()
                vc += (out.argmax(1) == lbls).sum().item()
                vt += lbls.size(0)

        ta = 100 * correct / total
        va = 100 * vc / vt
        current_lr = optimizer.param_groups[1]["lr"]

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Loss: {loss_sum/len(train_dl):.4f} | "
              f"Train: {ta:.1f}% | Val: {va:.1f}% | "
              f"LR: {current_lr:.2e}", flush=True)

        if va > best_acc:
            best_acc = va
            patience_counter = 0
            torch.save(model.state_dict(), "best_distraction_model_v2.pth")
            print(f"  → Saved best model ({va:.1f}%)")
        else:
            patience_counter += 1

        scheduler.step()

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Test with best model
    print("\n" + "=" * 50)
    model.load_state_dict(torch.load("best_distraction_model_v2.pth", map_location=device))
    model.eval()

    # Per-class accuracy
    class_correct = [0] * 3
    class_total = [0] * 3
    tc, tt = 0, 0

    with torch.no_grad():
        for imgs, lbls in test_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            preds = out.argmax(1)
            tc += (preds == lbls).sum().item()
            tt += lbls.size(0)
            for i in range(len(lbls)):
                label = lbls[i].item()
                class_total[label] += 1
                if preds[i].item() == label:
                    class_correct[label] += 1

    print(f"\nTest accuracy: {100*tc/tt:.1f}%")
    print(f"\nPer-class accuracy:")
    for i, cls in enumerate(train_ds.classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"  {cls:12s}: {acc:.1f}%  ({class_correct[i]}/{class_total[i]})")

    print(f"\nBest val accuracy: {best_acc:.1f}%")
    print(f"Model saved to: best_distraction_model_v2.pth")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(Path.home() / "distraction_dataset" / "splits"))
    p.add_argument("--checkpoint", default=str(Path.home() / "distraction_dataset" / "best_distraction_model.pth"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()
    main(args)
