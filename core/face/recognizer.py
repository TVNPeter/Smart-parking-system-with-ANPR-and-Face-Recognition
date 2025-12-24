from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from core.config import FACE_RECOGNITION_MODEL

try:
    import face_recognition  # type: ignore
except ImportError:
    face_recognition = None  # type: ignore


class FaceRecognizer:
    def __init__(self) -> None:
        self._available = face_recognition is not None
        self._model = FACE_RECOGNITION_MODEL
        if not self._available:
            print("Warning: face_recognition library not available. Please install it: pip install face_recognition")

    def extract(self, image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Extract face embedding and bounding box from BGR image.
        Returns: (embedding, bbox) where bbox is (x1, y1, x2, y2)
        """
        if not self._available or face_recognition is None:
            return None, None
        
        # Convert BGR to RGB (face_recognition uses RGB)
        image_rgb = image_bgr[:, :, ::-1]
        
        # Find face locations
        face_locations = face_recognition.face_locations(image_rgb, model=self._model)
        if not face_locations:
            return None, None
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
        if not face_encodings:
            return None, None
        
        # Select the largest face (by area)
        def area(loc):
            top, right, bottom, left = loc
            return (bottom - top) * (right - left)
        
        largest_idx = max(range(len(face_locations)), key=lambda i: area(face_locations[i]))
        
        # Get encoding and location
        encoding = face_encodings[largest_idx]
        top, right, bottom, left = face_locations[largest_idx]
        
        # Convert to (x1, y1, x2, y2) format
        bbox = (left, top, right, bottom)
        
        # Convert to numpy array with float32 dtype
        vec = np.asarray(encoding, dtype=np.float32)
        
        return vec, bbox
