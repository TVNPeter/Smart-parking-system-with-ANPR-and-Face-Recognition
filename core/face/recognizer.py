from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from core.config import INSIGHTFACE_MODEL, INSIGHTFACE_PROVIDER


class FaceRecognizer:
    def __init__(self) -> None:
        self._app = None
        self._ensure_app()

    def _ensure_app(self) -> None:
        if self._app is not None:
            return
        try:
            from insightface.app import FaceAnalysis  # type: ignore

            app = FaceAnalysis(name=INSIGHTFACE_MODEL, providers=[INSIGHTFACE_PROVIDER])
            app.prepare(ctx_id=0, det_size=(640, 640))
            self._app = app
        except Exception:
            self._app = None

    def extract(self, image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        if self._app is None:
            return None, None
        faces = self._app.get(image_bgr)
        if not faces:
            return None, None
        faces.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
        f0 = faces[0]
        emb = getattr(f0, "embedding", None)
        if emb is None and hasattr(f0, "normed_embedding"):
            emb = getattr(f0, "normed_embedding")
        if emb is None:
            return None, None
        vec = np.asarray(emb, dtype=np.float32)
        bbox = tuple(int(x) for x in f0.bbox.astype(int).tolist())  # type: ignore
        return vec, (bbox[0], bbox[1], bbox[2], bbox[3])
