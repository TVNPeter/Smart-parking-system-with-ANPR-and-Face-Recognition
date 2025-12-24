from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from core.db.database import init_db, SessionLocal, ParkingSession
from core.decision.verifier import Verifier


app = FastAPI(title="Smart Parking Core", version="0.1.0")
verifier = Verifier()


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.post("/entry")
async def entry(
    plate_image: UploadFile = File(...),
    face_image: UploadFile = File(...)
) -> Dict[str, Any]:
    plate_data = await plate_image.read()
    face_data = await face_image.read()
    res = verifier.handle_entry(plate_data, face_data)
    return {"session_id": res.session_id, "plate_text": res.plate_text, "status": res.status}


@app.post("/exit")
async def exit(
    plate_image: UploadFile = File(...),
    face_image: UploadFile = File(...)
) -> Dict[str, Any]:
    plate_data = await plate_image.read()
    face_data = await face_image.read()
    res = verifier.handle_exit(plate_data, face_data)
    return {"approved": res.approved, "fee": res.fee, "similarity_score": res.similarity_score, "session_id": res.session_id, "status": res.status}


@app.get("/sessions")
def sessions() -> Dict[str, Any]:
    out = []
    with SessionLocal() as db:
        for s in db.query(ParkingSession).order_by(ParkingSession.time_in.desc()).limit(200):
            out.append(
                {
                    "session_id": s.session_id,
                    "plate_text": s.plate_text,
                    "time_in": s.time_in.isoformat(),
                    "time_out": s.time_out.isoformat() if s.time_out else None,
                    "status": s.status,
                    "similarity": s.similarity,
                }
            )
    return {"items": out}
