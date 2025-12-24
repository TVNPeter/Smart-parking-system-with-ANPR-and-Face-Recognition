from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from rapidfuzz import fuzz
from sqlalchemy.orm import Session

from core.config import PLATES_DIR, FACES_DIR
from core.db.database import ParkingSession


def _save_crop(image_bgr: np.ndarray, bbox: Optional[Tuple[int, int, int, int]], base_dir: Path, prefix: str) -> str:
    base_dir.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}_{uuid.uuid4().hex[:8]}.jpg"
    out_path = base_dir / name
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        h, w = image_bgr.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            cv2.imwrite(str(out_path), crop)
            return out_path.as_posix()
    cv2.imwrite(str(out_path), image_bgr)
    return out_path.as_posix()


class SessionManager:
    def __init__(self) -> None:
        pass

    def create_session(
        self,
        db: Session,
        plate_text: str,
        image_bgr: np.ndarray,
        plate_bbox: Optional[Tuple[int, int, int, int]],
        face_embedding: Optional[np.ndarray],
        time_in: Optional[datetime] = None,
    ) -> ParkingSession:
        ts = time_in or datetime.utcnow()
        sid = uuid.uuid4().hex
        plate_path = _save_crop(image_bgr, plate_bbox, PLATES_DIR, "plate")

        rec = ParkingSession(
            session_id=sid,
            plate_text=plate_text,
            plate_image_path=plate_path,
            time_in=ts,
            time_out=None,
            status="ACTIVE",
            similarity=None,
        )
        rec.set_embedding(face_embedding)
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec

    def save_face_image(self, image_bgr: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]]) -> str:
        return _save_crop(image_bgr, face_bbox, FACES_DIR, "face")

    def get_active_candidates(self, db: Session, plate_text: str, min_ratio: int = 70) -> List[Tuple[ParkingSession, int]]:
        q = db.query(ParkingSession).filter(ParkingSession.status == "ACTIVE")
        items: List[Tuple[ParkingSession, int]] = []
        for rec in q:
            ratio = fuzz.ratio(plate_text.upper(), (rec.plate_text or "").upper())
            if ratio >= min_ratio:
                items.append((rec, int(ratio)))
        items.sort(key=lambda x: x[1], reverse=True)
        return items

    def close_session(self, db: Session, rec: ParkingSession, time_out: Optional[datetime], similarity: Optional[float]) -> ParkingSession:
        rec.time_out = time_out or datetime.utcnow()
        rec.status = "CLOSED"
        rec.similarity = float(similarity) if similarity is not None else None
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec

    def flag_session(self, db: Session, rec: ParkingSession, similarity: Optional[float]) -> ParkingSession:
        rec.status = "FLAGGED"
        rec.similarity = float(similarity) if similarity is not None else None
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec
