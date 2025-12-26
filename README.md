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
│   ├── detector.py       # Plate localization via OCR-only
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

Desktop UI Options
- Streamlit: Fast to prototype in a browser, simple layout for 2x2 camera grid, easy deployment. However, low-latency multi-camera streaming and tight OpenCV threading can be tricky; best for dashboards and light interactivity.
- customtkinter: Native desktop UI on Windows, better control over threads and OpenCV windows, straightforward access to local cameras, and lower latency. Recommended for a 4-camera grid with real-time "Check In" (capture face + plate) and "Check Out" (verify + pricing).

Recommendation: Use customtkinter for a responsive, kiosk-style app with four camera tiles and buttons. The core provides frame-based methods `handle_entry_frame()` and `handle_exit_frame()` in `core/decision/verifier.py` to integrate directly with cv2 frames.

GUI (customtkinter)
- Run the desktop app:
	- `python main.py`
- In the app:
	- Set each tile's source (index like 0/1 or URL), choose role `Plate` or `Face` for at least one tile each.
	- Press `Check In` to create a session; `Check Out` to verify and compute fee.
- Uses `insightface` for face embeddings and PaddleOCR for plate OCR.

Camera Sources
- Local webcams (Windows): use indices `0`, `1`, `2`, ...; you can probe available indices with the helper script.
- IP cameras (RTSP): example `rtsp://user:pass@192.168.1.50:554/Streaming/Channels/101`.
- HTTP MJPEG: example `http://192.168.1.60/mjpeg` (if your camera exposes MJPEG).
- Video files: provide a file path like `e:/videos/test.mp4`.

List available camera indices:
- Run: `python core/ui/list_cameras.py --max 10 --backend dshow`
- If none are found, try `--backend msmf` or `--backend default`.

Windows laptop camera tips
- Backend: `msmf` thường ổn định hơn `dshow` với camera tích hợp.
- Độ phân giải: ứng dụng sẽ yêu cầu 640x480 @30fps để tăng tốc khởi tạo.
- Codec: ứng dụng thử yêu cầu MJPG; nếu driver không hỗ trợ, sẽ dùng mặc định.
- Tránh trùng lặp: đảm bảo không app nào khác đang dùng camera (Teams/Zoom/etc.).

## Quickstart

### Cài đặt nhanh (Windows/Linux/Mac)

**1. Clone repository:**
```bash
git clone <repository-url>
cd Smart-parking-system-with-ANPR-and-Face-Recognition
```

**2. Tạo virtual environment:**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

**3. Cài đặt dependencies:**

**Option A: GPU Support (CUDA) - Khuyến nghị nếu có GPU:**
```bash
pip install -r requirements.txt
# requirements.txt đã bao gồm onnxruntime-gpu
```

**Option B: CPU Only (nếu không có GPU):**
```bash
# Sửa requirements.txt: comment onnxruntime-gpu, uncomment onnxruntime
pip install -r requirements.txt
```

**4. Chạy ứng dụng Desktop GUI:**
```bash
python main.py
```

**5. Hoặc chạy API server:**
```bash
uvicorn core.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Lưu ý về GPU (CUDA)

- **Nếu có GPU NVIDIA**: Hệ thống tự động phát hiện và sử dụng GPU
- **Nếu không có GPU**: Hệ thống tự động chuyển sang CPU
- **Để ép dùng GPU**: Set biến môi trường `INSIGHTFACE_PROVIDER=CUDAExecutionProvider`
- **Cần cài CUDA Toolkit và cuDNN** nếu muốn dùng GPU (xem chi tiết trong code comments)

Notes on Models
- Plate detection: Demo uses PaddleOCR text boxes only (no detector model).
- Face: insightface model family buffalo_l is used by default, running on CPU.

Pricing
Per the spec:
- duration_minutes = ceil((time_out - time_in) / 60)
- fee = duration_minutes * PRICE_PER_HOUR

This effectively charges per minute using the PRICE_PER_HOUR value. Adjust PRICE_PER_HOUR to your preferred per-minute rate for demos, or change the logic in core/session/pricing.py.

## Configuration

Environment variables (optional):
- `DATA_DIR` (default: `data`) - Thư mục lưu database và ảnh
- `FACE_THRESHOLD` (default: `0.38`) - Ngưỡng độ tương đồng khuôn mặt (0.0-1.0)
- `PRICE_PER_HOUR` (default: `2000`) - Giá đỗ xe mỗi giờ (VND)
- `PLATE_REGEX` (default: `[0-9]{2}[A-Z]?[\-\s]?[A-Z]?[0-9]{4,6}`) - Regex định dạng biển số VN
- `INSIGHTFACE_MODEL` (default: `buffalo_l`) - Mô hình InsightFace
- `INSIGHTFACE_PROVIDER` (default: auto-detect) - `CUDAExecutionProvider` hoặc `CPUExecutionProvider`

Tạo file `.env` trong thư mục gốc để cấu hình:
```env
FACE_THRESHOLD=0.38
PRICE_PER_HOUR=2000
INSIGHTFACE_PROVIDER=CUDAExecutionProvider
```

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
- Add PostgreSQL option for production
- Add basic auth / API keys if needed
 - Add customtkinter-based desktop UI (4-camera grid, actions)
