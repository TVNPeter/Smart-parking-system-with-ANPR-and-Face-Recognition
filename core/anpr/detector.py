from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

from core.anpr.ocr import PlateOCR


class LicensePlateDetector:
    def __init__(self, ocr: PlateOCR, model_path: Optional[str] = None) -> None:
        # YOLO support removed; detector now runs OCR-only on the full image.
        self.ocr = ocr

    def detect_plate(self, image_bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
        """Detect plate by running OCR on the full image."""
        text, bbox = self.ocr.recognize(image_bgr)
        return bbox, text
