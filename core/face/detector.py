from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from core.face.recognizer import FaceRecognizer


class FaceDetector:
    def __init__(self, recognizer: FaceRecognizer) -> None:
        self.recognizer = recognizer

    def detect(self, image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        emb, bbox = self.recognizer.extract(image_bgr)
        return bbox
