from pathlib import Path
import os


DATA_DIR = Path(os.getenv("DATA_DIR", "data")).resolve()
PLATES_DIR = DATA_DIR / "plates"
FACES_DIR = DATA_DIR / "faces"
DB_PATH = DATA_DIR / "app.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PLATES_DIR.mkdir(parents=True, exist_ok=True)
FACES_DIR.mkdir(parents=True, exist_ok=True)


FACE_THRESHOLD = float(os.getenv("FACE_THRESHOLD", "0.38"))
PRICE_PER_HOUR = float(os.getenv("PRICE_PER_HOUR", "2.0"))

# Vietnam plate format: 49-E122222, 30A-12345, etc.
PLATE_REGEX = os.getenv("PLATE_REGEX", r"[0-9]{2}[A-Z]?[\-\s]?[A-Z]?[0-9]{4,6}")
PLATE_MIN_CONF = float(os.getenv("PLATE_MIN_CONF", "0.2"))

YOLO_PLATE_MODEL = os.getenv("YOLO_PLATE_MODEL", "")
INSIGHTFACE_MODEL = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")
INSIGHTFACE_PROVIDER = os.getenv("INSIGHTFACE_PROVIDER", "CPUExecutionProvider")
