AI-based Smart Parking System (Core Software)

This project implements the core AI decision logic and API for a smart parking application. It focuses on session-based tracking, license plate OCR, face verification, hybrid AI + rules, and transparent pricing.

Features
- Session-based entry/exit flow with license plate as the primary key
- PaddleOCR for plate text (English); regex filtering for plate format
- InsightFace (RetinaFace + ArcFace) for face detection and 512-D embeddings
- Threshold-based cosine similarity verification (default 0.38)
- Time-based fee calculation (spec-defined formula)
- FastAPI backend with SQLite (dev) via SQLAlchemy

API
- POST /entry → image multipart
	- Output: session_id, plate_text, status
- POST /exit → image multipart
	- Output: approved, fee, similarity_score, session_id, status
- GET /sessions → monitoring list

Project Structure
core/
├── anpr/
│   ├── detector.py       # Plate localization via OCR (swap for YOLO later)
│   └── ocr.py            # PaddleOCR wrapper
├── face/
│   ├── detector.py       # Thin detector wrapper using insightface
│   └── recognizer.py     # 512-D embedding extractor
├── session/
│   ├── manager.py        # Session CRUD, fuzzy retrieval, image saving
│   └── pricing.py        # Pricing per spec
├── decision/
│   └── verifier.py       # Entry/exit pipeline + decision logic
├── db/
│   └── database.py       # SQLAlchemy ORM + init
└── api/
		└── main.py           # FastAPI app

Quickstart (Windows)
1) Create a virtual environment and install dependencies:
		python -m venv .venv
		.venv\Scripts\activate
		pip install -r requirements.txt

2) Optional: Configure environment in .env (see .env.example). Default FACE_THRESHOLD=0.38.

3) Run the server:
		uvicorn core.api.main:app --reload --host 0.0.0.0 --port 8000

4) Test with an image:
		curl -X POST http://localhost:8000/entry -F "image=@sample_entry.jpg"
		curl -X POST http://localhost:8000/exit -F "image=@sample_exit.jpg"

Notes on Models
- Plate detection: The current demo localizes plates using PaddleOCR text boxes. You can later plug in a YOLOv12-Nano/Small detector by replacing core/anpr/detector.py.
- Face: insightface model family buffalo_l is used by default, running on CPU.

Pricing
Per the spec:
- duration_minutes = ceil((time_out - time_in) / 60)
- fee = duration_minutes * PRICE_PER_HOUR

This effectively charges per minute using the PRICE_PER_HOUR value. Adjust PRICE_PER_HOUR to your preferred per-minute rate for demos, or change the logic in core/session/pricing.py.

Configuration
Environment variables (optional):
- DATA_DIR (default: data)
- FACE_THRESHOLD (default: 0.38)
- PRICE_PER_HOUR (default: 2.0)
- PLATE_REGEX (default: [A-Z0-9\-]{5,10})
- YOLO_PLATE_MODEL (optional; if you add a YOLO detector)
- INSIGHTFACE_MODEL (default: buffalo_l)
- INSIGHTFACE_PROVIDER (default: CPUExecutionProvider)

Storage
- SQLite DB at data/app.db
- Cropped plate images at data/plates/
- Cropped face images at data/faces/

Non-goals
- No training
- No hardware control
- No cloud dependency
- No retention beyond embeddings; embeddings are stored in DB as blobs

Roadmap hooks
- Swap in YOLOv12 detector for plates
- Add PostgreSQL option for production
- Add basic auth / API keys if needed
