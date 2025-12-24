from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy import (
    String,
    DateTime,
    create_engine,
    LargeBinary,
    Float,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from core.config import DB_PATH


class Base(DeclarativeBase):
    pass


class ParkingSession(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    plate_text: Mapped[str] = mapped_column(String(32), index=True)
    plate_image_path: Mapped[str] = mapped_column(String(512))

    face_embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    embedding_shape: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    embedding_dtype: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)

    time_in: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)
    time_out: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True, index=True)

    status: Mapped[str] = mapped_column(String(16), index=True)
    similarity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    def get_embedding(self) -> Optional[np.ndarray]:
        if self.face_embedding is None or self.embedding_shape is None or self.embedding_dtype is None:
            return None
        arr = np.frombuffer(self.face_embedding, dtype=np.dtype(self.embedding_dtype))
        shape = tuple(int(x) for x in self.embedding_shape.split(","))
        return arr.reshape(shape)

    def set_embedding(self, emb: Optional[np.ndarray]) -> None:
        if emb is None:
            self.face_embedding = None
            self.embedding_shape = None
            self.embedding_dtype = None
            return
        self.face_embedding = emb.astype(np.float32).tobytes()
        self.embedding_shape = ",".join(str(x) for x in emb.shape)
        self.embedding_dtype = str(np.float32().dtype)


def _sqlite_url(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


engine = create_engine(_sqlite_url(DB_PATH), future=True, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.commit()
