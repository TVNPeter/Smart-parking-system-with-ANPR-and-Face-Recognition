from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import cv2
import numpy as np

from core.anpr.detector import LicensePlateDetector
from core.anpr.ocr import PlateOCR
from core.config import FACE_THRESHOLD, PRICE_PER_HOUR, YOLO_PLATE_MODEL
from core.db.database import SessionLocal
from core.face.recognizer import FaceRecognizer
from core.session.manager import SessionManager
from core.session.pricing import compute_fee


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))


@dataclass
class EntryResult:
    session_id: str
    plate_text: str
    status: str


@dataclass
class ExitResult:
    approved: bool
    fee: float
    similarity_score: float
    session_id: Optional[str]
    status: str


class Verifier:
    def __init__(self) -> None:
        self.ocr = PlateOCR()
        self.detector = LicensePlateDetector(self.ocr, model_path=YOLO_PLATE_MODEL if YOLO_PLATE_MODEL else None)
        self.face = FaceRecognizer()
        self.sessions = SessionManager()

    def _imread_bgr(self, data: bytes) -> np.ndarray:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    def handle_entry(self, plate_image_bytes: bytes, face_image_bytes: bytes) -> EntryResult:
        plate_img = self._imread_bgr(plate_image_bytes)
        face_img = self._imread_bgr(face_image_bytes)
        
        plate_bbox, plate_text = self.detector.detect_plate(plate_img)
        face_emb, _ = self.face.extract(face_img)

        with SessionLocal() as db:
            rec = self.sessions.create_session(
                db, 
                plate_text=plate_text or "", 
                image_bgr=plate_img, 
                plate_bbox=plate_bbox, 
                face_embedding=face_emb, 
                time_in=datetime.utcnow()
            )
            return EntryResult(session_id=rec.session_id, plate_text=rec.plate_text, status=rec.status)

    def handle_exit(self, plate_image_bytes: bytes, face_image_bytes: bytes, face_threshold: float = FACE_THRESHOLD) -> ExitResult:
        plate_img = self._imread_bgr(plate_image_bytes)
        face_img = self._imread_bgr(face_image_bytes)
        
        _, plate_text = self.detector.detect_plate(plate_img)
        face_emb, _ = self.face.extract(face_img)

        with SessionLocal() as db:
            cands = self.sessions.get_active_candidates(db, plate_text or "")
            if not cands:
                return ExitResult(approved=False, fee=0.0, similarity_score=0.0, session_id=None, status="FLAGGED")

            best_sim = -1.0
            best_rec = None

            for rec, _ratio in cands:
                emb = rec.get_embedding()
                if face_emb is None or emb is None:
                    sim = -1.0
                else:
                    sim = _cosine_similarity(face_emb, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_rec = rec

            if best_rec is None:
                return ExitResult(approved=False, fee=0.0, similarity_score=0.0, session_id=None, status="FLAGGED")

            now = datetime.utcnow()
            if best_sim >= face_threshold:
                closed = self.sessions.close_session(db, best_rec, time_out=now, similarity=best_sim)
                fee = compute_fee(closed.time_in, closed.time_out or now, PRICE_PER_HOUR)
                return ExitResult(approved=True, fee=fee, similarity_score=best_sim, session_id=closed.session_id, status=closed.status)

            flagged = self.sessions.flag_session(db, best_rec, similarity=best_sim)
            return ExitResult(approved=False, fee=0.0, similarity_score=best_sim, session_id=flagged.session_id, status=flagged.status)
