#!/usr/bin/env python3
"""
Live webcam inference with the trained distraction detector.
Includes phone detection overlay when distracted.

Usage:
  python run_live.py                    # webcam live feed
  python run_live.py --image photo.jpg  # single image
  python run_live.py --no-phone         # disable phone detection
  python run_live.py --help
"""
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
from pathlib import Path
from ultralytics import YOLO

CLASSES = ["distracted", "empty", "focused"]
COLORS = {
    "focused":    (0, 200, 0),     # green
    "distracted": (0, 0, 220),     # red
    "empty":      (200, 180, 0),   # cyan
}
# COCO class ID 67 = "cell phone"
PHONE_CLASS_ID = 67

MODEL_PATH = Path(__file__).parent / "best_distraction_model.pth"

def load_model(path):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 3)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def load_phone_detector():
    """Load YOLOv8-nano for phone detection."""
    detector = YOLO("yolov8n.pt")
    return detector

def detect_phones(detector, frame, conf_thresh=0.35):
    """Detect cell phones in frame. Returns list of (x1, y1, x2, y2, conf)."""
    results = detector(frame, classes=[PHONE_CLASS_ID], conf=conf_thresh, verbose=False)
    phones = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            phones.append((x1, y1, x2, y2, conf))
    return phones

def draw_phone_boxes(frame, phones):
    """Draw pulsing bounding boxes around detected phones."""
    for (x1, y1, x2, y2, conf) in phones:
        # Red bounding box with thick border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # Semi-transparent red overlay on the phone region
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        # Label above box
        label = f"PHONE {conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), (0, 0, 180), -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(model, frame):
    """Returns (label, confidence, all_probs)"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = transform(rgb).unsqueeze(0)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1)[0].numpy()
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]) * 100, probs

def draw_overlay(frame, label, conf, probs, phone_detected=False):
    h, w = frame.shape[:2]
    color = COLORS[label]

    # Border
    cv2.rectangle(frame, (0, 0), (w-1, h-1), color, 8)

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)
    cv2.putText(frame, f"{label.upper()}  {conf:.1f}%",
                (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    # Phone warning badge next to label when distracted + phone found
    if label == "distracted" and phone_detected:
        cv2.putText(frame, "PHONE DETECTED",
                    (w - 260, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Probability bars
    bar_x = w - 280
    for i, cls in enumerate(CLASSES):
        y = 90 + i * 30
        bar_w = int(probs[i] * 220)
        cv2.rectangle(frame, (bar_x, y), (bar_x + 220, y + 20), (40, 40, 40), -1)
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + 20), COLORS[cls], -1)
        cv2.putText(frame, f"{cls}: {probs[i]*100:.1f}%",
                    (bar_x + 4, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Controls
    cv2.putText(frame, "Q = quit  |  S = screenshot",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
    return frame

def run_live(model, phone_detector=None):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("\n  LIVE DISTRACTION DETECTOR")
    print("  ─────────────────────────")
    print("  Q = quit")
    print("  S = save screenshot")
    if phone_detector:
        print("  Phone detection: ON")
    print()

    snap_count = 0
    history = []
    frame_count = 0
    cached_phones = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, conf, probs = predict(model, frame)

        # Smooth over last 5 frames
        history.append(probs.copy())
        if len(history) > 5:
            history.pop(0)
        avg = np.mean(history, axis=0)
        idx = int(np.argmax(avg))
        label = CLASSES[idx]
        conf = float(avg[idx]) * 100

        # Phone detection (only when distracted, every 3rd frame for speed)
        phone_detected = False
        if phone_detector and label == "distracted":
            frame_count += 1
            if frame_count % 3 == 0 or not cached_phones:
                cached_phones = detect_phones(phone_detector, frame)
            if cached_phones:
                phone_detected = True
                frame = draw_phone_boxes(frame, cached_phones)
        else:
            cached_phones = []

        frame = draw_overlay(frame, label, conf, avg, phone_detected)
        cv2.imshow("Distraction Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            snap_count += 1
            path = f"screenshot_{snap_count:03d}_{label}.jpg"
            cv2.imwrite(path, frame)
            print(f"  Saved: {path}")

    cap.release()
    cv2.destroyAllWindows()

def run_image(model, path, phone_detector=None):
    frame = cv2.imread(path)
    if frame is None:
        print(f"ERROR: Could not read {path}")
        return
    label, conf, probs = predict(model, frame)
    print(f"\n  Prediction: {label.upper()} ({conf:.1f}%)")
    for i, cls in enumerate(CLASSES):
        print(f"    {cls}: {probs[i]*100:.1f}%")

    # Phone detection for distracted images
    phone_detected = False
    if phone_detector and label == "distracted":
        phones = detect_phones(phone_detector, frame)
        if phones:
            phone_detected = True
            frame = draw_phone_boxes(frame, phones)
            print(f"\n  Phones detected: {len(phones)}")

    frame = draw_overlay(frame, label, conf, probs, phone_detected)
    cv2.imshow("Result", frame)
    print("\n  Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    p = argparse.ArgumentParser(description="Run distraction detector")
    p.add_argument("--image", type=str, help="Path to a single image (skip webcam)")
    p.add_argument("--model", type=str, default=str(MODEL_PATH), help="Path to .pth model")
    p.add_argument("--no-phone", action="store_true", help="Disable phone detection overlay")
    args = p.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(args.model)
    print("Model loaded.")

    phone_detector = None
    if not args.no_phone:
        print("Loading phone detector (YOLOv8n)...")
        phone_detector = load_phone_detector()
        print("Phone detector loaded.\n")
    else:
        print("Phone detection: disabled\n")

    if args.image:
        run_image(model, args.image, phone_detector)
    else:
        run_live(model, phone_detector)

if __name__ == "__main__":
    main()
