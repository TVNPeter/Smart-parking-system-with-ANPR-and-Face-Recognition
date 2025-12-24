from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

from core.anpr.ocr import PlateOCR


class LicensePlateDetector:
    def __init__(self, ocr: PlateOCR, model_path: Optional[str] = None) -> None:
        self.ocr = ocr
        self._yolo = None
        self._use_yolo = False
        self._init_yolo(model_path)

    def _init_yolo(self, model_path: Optional[str] = None) -> None:
        """Initialize YOLO model for plate detection."""
        if model_path is None:
            # Try default location
            model_path = os.path.join(os.getcwd(), "models", "plate_detector.pt")
        
        model_path = model_path.strip() if isinstance(model_path, str) else ""
        if not model_path or not os.path.exists(model_path):
            return
        
        try:
            from ultralytics import YOLO  # type: ignore
            self._yolo = YOLO(model_path)
            self._use_yolo = True
            print(f"[ANPR] YOLO detector loaded: {model_path}")
        except Exception as e:
            print(f"[ANPR] YOLO init failed: {e}, falling back to OCR-only")
            self._yolo = None
            self._use_yolo = False

    def detect_plate(self, image_bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
        """
        Detect license plate:
        1. If YOLO available: locate plate region → crop → OCR on crop
        2. Otherwise: OCR on full image
        """
        # Try YOLO-based detection
        if self._use_yolo and self._yolo is not None:
            try:
                results = self._yolo.predict(image_bgr, verbose=False)
                if results and results[0].boxes is not None:
                    xyxy = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    
                    if len(xyxy) > 0:
                        # Pick highest confidence box
                        best_idx = np.argmax(confs)
                        x1, y1, x2, y2 = map(int, xyxy[best_idx])
                        
                        # Clamp to image bounds
                        h, w = image_bgr.shape[:2]
                        x1 = max(0, min(x1, w - 1))
                        x2 = max(x1 + 1, min(x2, w))
                        y1 = max(0, min(y1, h - 1))
                        y2 = max(y1 + 1, min(y2, h))
                        
                        # Crop and OCR
                        roi = image_bgr[y1:y2, x1:x2]
                        text, _ = self.ocr.recognize(roi)
                        return (x1, y1, x2, y2), text
            except Exception as e:
                print(f"[ANPR] YOLO detection failed: {e}, falling back to OCR")
        
        # Fallback: OCR on full image
        text, bbox = self.ocr.recognize(image_bgr)
        return bbox, text
