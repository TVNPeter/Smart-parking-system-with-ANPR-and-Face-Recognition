from __future__ import annotations

import re
import os
from typing import Optional, Tuple, List

import numpy as np

from core.config import PLATE_REGEX, PLATE_MIN_CONF

DEBUG = os.getenv("DEBUG_OCR", "0") == "1"


class PlateOCR:
    def __init__(self) -> None:
        self._paddle = None
        self._try_init_paddle()

    def _try_init_paddle(self) -> None:
        try:
            from paddleocr import PaddleOCR  # type: ignore

            self._paddle = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except Exception:
            self._paddle = None

    def recognize(self, image_bgr: np.ndarray) -> Tuple[str, Optional[Tuple[int, int, int, int]]]:
        if self._paddle is None:
            return "", None

        result = self._paddle.ocr(image_bgr, cls=True)
        if not result or not result[0]:
            if DEBUG:
                print("[OCR] No text detected")
            return "", None

        if DEBUG:
            print(f"[OCR] Raw results: {result[0]}")

        best_text = ""
        best_conf = -1.0
        best_box: Optional[List[List[float]]] = None
        plate_rx = re.compile(PLATE_REGEX, re.IGNORECASE)

        # First pass: try to match plate pattern
        for line in result[0]:
            box, (text, conf) = line
            if not text:
                continue
            clean_text = text.upper().strip().replace(" ", "").replace("O", "0")
            if DEBUG:
                print(f"[OCR] Text: '{text}' -> '{clean_text}' (conf={conf:.2f})")
            
            if plate_rx.search(clean_text) and conf >= PLATE_MIN_CONF:
                if conf > best_conf:
                    best_text = clean_text
                    best_conf = conf
                    best_box = box
                    if DEBUG:
                        print(f"[OCR] ✓ Matched plate pattern: {best_text}")

        # Second pass: if no match, take highest confidence text
        if not best_text:
            if DEBUG:
                print("[OCR] No regex match, using highest confidence")
            for line in result[0]:
                box, (text, conf) = line
                if conf > best_conf:
                    best_text = text.upper().strip().replace(" ", "").replace("O", "0")
                    best_conf = conf
                    best_box = box

        bbox = None
        if best_box is not None:
            xs = [p[0] for p in best_box]
            ys = [p[1] for p in best_box]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            bbox = (x1, y1, x2, y2)

        if DEBUG:
            print(f"[OCR] Final result: '{best_text}' (conf={best_conf:.2f})")
        
        return best_text, bbox
