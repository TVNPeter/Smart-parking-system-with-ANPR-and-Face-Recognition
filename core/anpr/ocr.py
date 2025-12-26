from __future__ import annotations

import re
import os
from typing import Optional, Tuple, List, Any

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
            # paddleocr>=3.3 removed show_log kwarg; rely on default logging settings.
            # use_gpu=True enables CUDA acceleration if available
            self._paddle = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True)
        except Exception as e:
            self._paddle = None
            if DEBUG:
                print(f"[OCR] PaddleOCR init failed: {e.__class__.__name__}: {e}")

    def recognize(self, image_bgr: np.ndarray) -> Tuple[str, Optional[Tuple[int, int, int, int]]]:
        if self._paddle is None:
            return "", None

        # paddleocr>=3.3 pipelines ignore/forbid cls kwarg; angle cls is enabled via constructor.
        result = self._paddle.ocr(image_bgr)
        if not result:
            if DEBUG:
                print("[OCR] No text detected")
            return "", None

        # paddleocr 3.3+ returns a dict (doc pipeline) instead of list of lines.
        lines: List[Any] = []
        first = result[0] if isinstance(result, list) else result

        if isinstance(first, dict):
            texts = first.get("rec_texts", []) or []
            scores = first.get("rec_scores", []) or []
            boxes = first.get("rec_polys") or first.get("dt_polys") or []
            for i, text in enumerate(texts):
                score = scores[i] if i < len(scores) else 0.0
                box = boxes[i] if i < len(boxes) else None
                if box is None:
                    continue
                # Normalize to list-of-points
                box_list = np.array(box).tolist()
                lines.append([box_list, (text, score)])
        elif isinstance(first, list):
            # Legacy format already compatible: list of [box, (text, conf)]
            lines = first
        else:
            if DEBUG:
                print(f"[OCR] Unsupported result type: {type(first)}")
            return "", None

        if DEBUG:
            print(f"[OCR] Raw results: {first}")

        best_text = ""
        best_conf = -1.0
        best_box: Optional[List[List[float]]] = None
        plate_rx = re.compile(PLATE_REGEX, re.IGNORECASE)

        # Normalize per-line entries and cache for combination
        norm_lines = []
        for line in lines:
            box, (text, conf) = line
            if not text:
                continue
            clean_text = (
                text.upper()
                .strip()
                .replace(" ", "")
                .replace("-", "")
                .replace(".", "")
                .replace("O", "0")
            )
            y_center = float(np.mean([p[1] for p in box])) if box else 0.0
            norm_lines.append((box, text, clean_text, conf, y_center))
            if DEBUG:
                print(f"[OCR] Text: '{text}' -> '{clean_text}' (conf={conf:.2f})")

        # First pass: regex on individual lines
        for box, text, clean_text, conf, _ in norm_lines:
            if plate_rx.search(clean_text) and conf >= PLATE_MIN_CONF:
                if conf > best_conf:
                    best_text = clean_text
                    best_conf = conf
                    best_box = box
                    if DEBUG:
                        print(f"[OCR] ✓ Matched plate pattern: {best_text}")

        # Second pass: try combining top-to-bottom lines (Vietnam plates often 2 lines)
        if not best_text and norm_lines:
            norm_lines_sorted = sorted(norm_lines, key=lambda x: x[4])  # sort by y_center
            combined_text = "".join(item[2] for item in norm_lines_sorted)
            combined_conf = float(np.mean([item[3] for item in norm_lines_sorted]))
            combined_box_pts: List[List[float]] = []
            for box, *_ in norm_lines_sorted:
                combined_box_pts.extend(box)
            if plate_rx.search(combined_text) and combined_conf >= PLATE_MIN_CONF:
                best_text = combined_text
                best_conf = combined_conf
                best_box = combined_box_pts
                if DEBUG:
                    print(f"[OCR] ✓ Combined lines matched: {best_text} (conf~{combined_conf:.2f})")

        # Third pass: highest confidence fallback
        if not best_text:
            if DEBUG:
                print("[OCR] No regex match, using highest confidence")
            for box, text, clean_text, conf, _ in norm_lines:
                if conf > best_conf:
                    best_text = clean_text
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
