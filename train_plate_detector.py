"""
Train YOLOv8 detector for Vietnamese license plates.

Prerequisites:
  1. Prepare dataset in YOLO format:
     dataset/
     ├── images/
     │   ├── train/
     │   ├── val/
     │   └── test/
     └── labels/
         ├── train/
         ├── val/
         └── test/

  2. Create data.yaml:
     path: /path/to/dataset
     train: images/train
     val: images/val
     test: images/test
     nc: 1
     names: ['plate']

  3. Install dependencies:
     pip install ultralytics opencv-python

  4. Run training:
     python train_plate_detector.py --data data.yaml --epochs 100 --imgsz 640
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def train_yolo(data_yaml: str, epochs: int = 100, imgsz: int = 640, device: str = "cpu"):
    """Train YOLOv8 for plate detection."""
    
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"❌ Error: data.yaml not found at {data_yaml}")
        sys.exit(1)
    
    print(f"📦 Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")  # nano model (lightweight)
    
    print(f"🚀 Starting training...")
    print(f"   Data: {data_yaml}")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Device: {device}")
    
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        patience=20,  # Early stopping
        save=True,
        project="models",
        name="plate_detector",
        exist_ok=True,
        verbose=True,
    )
    
    print(f"\n✅ Training complete!")
    print(f"📊 Results saved to: models/plate_detector/")
    print(f"💾 Best model: models/plate_detector/weights/best.pt")
    print(f"\n💡 To use in production:")
    print(f"   export YOLO_PLATE_MODEL=models/plate_detector/weights/best.pt")
    print(f"   python -m uvicorn core.api.main:app --reload")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for plate detection")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    
    args = parser.parse_args()
    train_yolo(args.data, args.epochs, args.imgsz, args.device)


if __name__ == "__main__":
    main()
