"""Quick OCR-only test on plate.jpg (no YOLO)"""
import argparse
import cv2
from pathlib import Path

from core.anpr.ocr import PlateOCR


def parse_args():
    p = argparse.ArgumentParser(description="OCR-only license plate test")
    p.add_argument("--image", default="plate.jpg", help="Path to plate image (BGR)")
    return p.parse_args()


def main():
    args = parse_args()
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"❌ Image not found: {img_path}")
        return 1

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Failed to read image: {img_path}")
        return 1

    ocr = PlateOCR()
    text, bbox = ocr.recognize(img)
    print("📋 OCR-only result:")
    print(f"   Text: '{text}'")
    print(f"   Bbox: {bbox}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
