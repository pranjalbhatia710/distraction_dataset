#!/usr/bin/env python3
"""
Train distraction detection model using MobileNetV3 (transfer learning).
Usage: python train_model.py --data ~/distraction_dataset/splits --epochs 20
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = datasets.ImageFolder(Path(args.data)/"train", train_tf)
    val_ds = datasets.ImageFolder(Path(args.data)/"val", val_tf)
    test_ds = datasets.ImageFolder(Path(args.data)/"test", val_tf)

    print(f"Classes: {train_ds.classes}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=32, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=32, num_workers=4)

    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        loss_sum, correct, total = 0, 0, 0
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            correct += (out.argmax(1)==lbls).sum().item()
            total += lbls.size(0)

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_dl:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                vc += (out.argmax(1)==lbls).sum().item()
                vt += lbls.size(0)

        ta, va = 100*correct/total, 100*vc/vt
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss_sum/len(train_dl):.4f} | Train: {ta:.1f}% | Val: {va:.1f}%")

        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), "best_distraction_model.pth")
            print(f"  → Saved best model ({va:.1f}%)")
        scheduler.step()

    # Test
    model.load_state_dict(torch.load("best_distraction_model.pth"))
    model.eval()
    tc, tt = 0, 0
    with torch.no_grad():
        for imgs, lbls in test_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            tc += (out.argmax(1)==lbls).sum().item()
            tt += lbls.size(0)
    print(f"\nTest accuracy: {100*tc/tt:.1f}%")
    print("Model saved to: best_distraction_model.pth")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(Path.home()/"distraction_dataset"/"splits"))
    p.add_argument("--epochs", type=int, default=20)
    main(p.parse_args())
