"""Quick test: OCR on plate.jpg to verify pipeline"""
import os
import cv2
from core.anpr.ocr import PlateOCR
from core.anpr.detector import LicensePlateDetector

os.environ["DEBUG_OCR"] = "1"  # Enable debug output

if __name__ == "__main__":
    # Load image
    plate_img = cv2.imread("plate.jpg")
    if plate_img is None:
        print("❌ Cannot read plate.jpg")
        exit(1)
    
    print(f"📸 Image shape: {plate_img.shape}")
    
    # Test OCR only
    ocr = PlateOCR()
    text, bbox = ocr.recognize(plate_img)
    print(f"\n📋 OCR Result:")
    print(f"   Text: '{text}'")
    print(f"   Bbox: {bbox}")
    
    # Test detector (OCR-only for now)
    detector = LicensePlateDetector(ocr)
    bbox, plate_text = detector.detect_plate(plate_img)
    print(f"\n🎯 Detector Result:")
    print(f"   Plate text: '{plate_text}'")
    print(f"   Bbox: {bbox}")
