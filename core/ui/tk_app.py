from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime
from typing import Optional, Tuple, List

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image

from core.decision.verifier import Verifier
from core.db.database import init_db


# OpenCV backends (Windows)
BACKENDS = {
    "auto": 0,  # let OpenCV choose
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}


class CameraWorker:
    def __init__(self) -> None:
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.source_str: Optional[str] = None
        self.backend_name: str = "auto"
        self.status: str = "idle"  # idle, opening, running, error

    def start(self, source: str, backend_name: str = "auto") -> None:
        # Set parameters and start background loop; avoid blocking UI on open
        self.stop()
        self.source_str = source
        self.backend_name = backend_name
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while self.running:
            # Lazily open the capture inside the background thread
            if self.cap is None:
                src = self.source_str
                if not src:
                    time.sleep(0.05)
                    continue
                
                self.status = "opening"
                # parse index if possible
                idx: Optional[int] = None
                try:
                    idx = int(src)
                except Exception:
                    idx = None
                
                # Detect if source is an image file (not video/camera)
                is_image = False
                if idx is None:
                    lower_src = src.lower()
                    if any(lower_src.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"]):
                        is_image = True
                
                # IMAGE: use imread instead of VideoCapture
                if is_image:
                    try:
                        img = cv2.imread(src)
                        if img is not None:
                            # Wrap as single-frame "capture"
                            with self.lock:
                                self.latest_frame = img
                            self.status = "running"
                            # Keep serving this frame repeatedly
                            while self.running:
                                time.sleep(0.1)
                            return
                    except Exception:
                        pass
                    self.status = "error"
                    time.sleep(0.2)
                    continue
                
                # CAMERA/VIDEO: use VideoCapture
                # Only use backend flag for integer indices (MSMF/DSHOW don't support string paths)
                if idx is not None:
                    flag = BACKENDS.get(self.backend_name, 0)
                    try:
                        cap = cv2.VideoCapture(idx, flag) if flag != 0 else cv2.VideoCapture(idx)
                    except Exception:
                        cap = cv2.VideoCapture(idx)
                else:
                    # For URLs/paths, use default backend
                    cap = cv2.VideoCapture(src)

                # Fast-start tuning: request modest resolution/FPS and MJPG
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                except Exception:
                    pass

                # Minimal verification for fast startup
                ok_open = cap.isOpened()
                ok_read = False
                if ok_open:
                    # For files/URLs: just check isOpened, skip frame read
                    # For cameras: try 1-2 quick reads only
                    if idx is not None:
                        for _ in range(2):
                            ok_read, frame = cap.read()
                            if ok_read and frame is not None:
                                break
                            time.sleep(0.02)
                    else:
                        # Files open instantly, assume OK if isOpened
                        ok_read = True
                
                if ok_open and ok_read:
                    self.cap = cap
                    self.status = "running"
                else:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    self.status = "error"
                    time.sleep(0.2)
                    continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            with self.lock:
                self.latest_frame = frame
            time.sleep(0.01)

    def get_latest(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self) -> None:
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.status = "idle"
        with self.lock:
            self.latest_frame = None


class CameraTile(ctk.CTkFrame):
    def __init__(self, master: ctk.CTk, name: str, fixed_role: Optional[str] = None) -> None:
        super().__init__(master)
        self.name = name
        self.fixed_role = fixed_role
        self.worker = CameraWorker()
        self.scanning = False
        self.frozen_frame: Optional[np.ndarray] = None  # Store frozen frame for Face IN camera
        label_text = f"{name} ({fixed_role})\n(no video)" if fixed_role else f"{name}\n(no video)"
        self.image_label = ctk.CTkLabel(self, text=label_text)
        self.image_label.grid(row=0, column=0, columnspan=4, padx=6, pady=6, sticky="nsew")

        self.source_entry = ctk.CTkEntry(self, placeholder_text="index (0/1/2) or rtsp/http/file path")
        self.source_entry.grid(row=1, column=0, columnspan=2, padx=6, pady=4, sticky="ew")
        self.backend_opt = ctk.CTkOptionMenu(self, values=["auto", "dshow", "msmf"])
        # Default to msmf for laptop cameras to avoid dshow stalls
        self.backend_opt.set("msmf")
        self.backend_opt.grid(row=1, column=2, padx=6, pady=4, sticky="ew")

        if fixed_role:
            self.role_opt = None
            self.role_label = ctk.CTkLabel(self, text=fixed_role, fg_color="gray20", corner_radius=6)
            self.role_label.grid(row=2, column=2, padx=6, pady=4, sticky="ew")
        else:
            self.role_label = None
            self.role_opt = ctk.CTkOptionMenu(self, values=["None", "Plate", "Face"])
            self.role_opt.set("None")
            self.role_opt.grid(row=2, column=2, padx=6, pady=4, sticky="ew")

        self.btn_start = ctk.CTkButton(self, text="Start", command=self._start)
        self.btn_stop = ctk.CTkButton(self, text="Stop", command=self._stop)
        self.btn_start.grid(row=1, column=3, padx=6, pady=4, sticky="ew")
        self.btn_stop.grid(row=2, column=3, padx=6, pady=4, sticky="ew")

        # Detected camera sources and scan
        self.detected_opt = ctk.CTkOptionMenu(self, values=["None"])
        self.detected_opt.set("None")
        self.detected_opt.grid(row=2, column=0, padx=6, pady=4, sticky="ew")
        self.btn_scan = ctk.CTkButton(self, text="Scan", command=self._scan)
        self.btn_scan.grid(row=2, column=1, padx=6, pady=4, sticky="ew")

        self.status = ctk.CTkLabel(self, text="idle")
        self.status.grid(row=3, column=0, columnspan=4, padx=6, pady=4, sticky="ew")

        # Configure grid weights for consistent sizing
        for i in range(4):
            self.grid_columnconfigure(i, weight=1, uniform="camera_col")
        # Row 0 (image) gets most space, rows 1-3 (controls) get minimal space
        self.grid_rowconfigure(0, weight=1, uniform="camera_row")
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.grid_rowconfigure(3, weight=0)

        self.after(100, self._update_image)  # Reduced from 50ms
        self.after(250, self._update_status)  # Reduced from 100ms

    def _update_status(self) -> None:
        # Poll worker status and update UI
        ws = self.worker.status
        if ws == "opening":
            self.status.configure(text="opening camera...")
        elif ws == "error":
            self.status.configure(text="failed to open")
        self.after(250, self._update_status)  # Reduced from 100ms

    def _start(self) -> None:
        # Prefer manual entry first, then detected selection
        src = self.source_entry.get().strip()
        if not src:
            # Only use detected if manual entry is empty
            det = self.detected_opt.get()
            if det and det != "None":
                if det.startswith("index "):
                    try:
                        parts = det.split()
                        idx = parts[1]
                        backend = parts[2].strip("()") if len(parts) > 2 else "auto"
                        src = idx
                        self.backend_opt.set(backend)
                    except Exception:
                        pass
        if not src:
            self.status.configure(text="Enter source index or URL")
            return
        try:
            self.worker.start(src, backend_name=self.backend_opt.get())
            self.status.configure(text=f"running: {src}")
        except Exception as e:
            self.status.configure(text=f"error: {e}")

    def _stop(self) -> None:
        self.worker.stop()
        self.status.configure(text="stopped")

    def _update_image(self) -> None:
        # Use frozen frame if available (for Face IN camera), otherwise use live frame
        if self.frozen_frame is not None:
            frame = self.frozen_frame
        else:
            frame = self.worker.get_latest()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            # Get label dimensions, ensure consistent sizing across all tiles
            label_w = self.image_label.winfo_width()
            label_h = self.image_label.winfo_height()
            if label_w > 1 and label_h > 1:
                # Use actual label size for consistent display
                w = label_w
                h = label_h
            else:
                # Fallback: use fixed aspect ratio (16:9) for consistency
                w = 400
                h = 300
            # Maintain aspect ratio
            frame_aspect = frame.shape[1] / frame.shape[0]
            label_aspect = w / h
            if frame_aspect > label_aspect:
                # Frame is wider, fit to width
                w = min(w, 480)
                h = int(w / frame_aspect)
            else:
                # Frame is taller, fit to height
                h = min(h, 360)
                w = int(h * frame_aspect)
            # Ensure minimum size
            w = max(160, w)
            h = max(120, h)
            img = img.resize((w, h), Image.Resampling.LANCZOS)
            cimg = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
            self.image_label.configure(image=cimg, text="")
            self.image_label.image = cimg
        self.after(100, self._update_image)  # Reduced from 50ms

    def get_role(self) -> str:
        if self.fixed_role:
            return self.fixed_role
        return self.role_opt.get() if self.role_opt else "None"

    def get_frame(self) -> Optional[np.ndarray]:
        # Return frozen frame if available, otherwise return live frame
        if self.frozen_frame is not None:
            return self.frozen_frame.copy()
        return self.worker.get_latest()
    
    def freeze_frame(self) -> None:
        """Freeze the current frame (used for Face IN camera to preserve check-in image)"""
        frame = self.worker.get_latest()
        if frame is not None:
            self.frozen_frame = frame.copy()
            self.status.configure(text="frozen (check-in captured)")
    
    def unfreeze_frame(self) -> None:
        """Unfreeze the frame (reset to live camera)"""
        self.frozen_frame = None
        self.status.configure(text="live")

    def _scan(self) -> None:
        # Run probing asynchronously to avoid blocking the UI thread
        if self.scanning:
            return
        self.scanning = True
        self.btn_scan.configure(state="disabled")
        # Force DSHOW for scanning (most reliable on Windows)
        backend = "dshow"
        self.status.configure(text=f"scanning ({backend})…")

        def _worker() -> None:
            flag = BACKENDS.get(backend, 0)
            found = []
            max_indices = 5
            for i in range(max_indices):
                try:
                    # Always use flag for integer index scanning
                    cap = cv2.VideoCapture(i, flag) if flag != 0 else cv2.VideoCapture(i)
                    ok_open = cap.isOpened()
                    # Skip frame read - just check if device opens
                    cap.release()
                    if ok_open:
                        found.append(f"index {i} ({backend})")
                except Exception:
                    pass

            def _apply() -> None:
                if not found:
                    self.detected_opt.configure(values=["None"])
                    self.detected_opt.set("None")
                    self.status.configure(text=f"no cameras with {backend}")
                else:
                    self.detected_opt.configure(values=found)
                    self.detected_opt.set(found[0])
                    self.status.configure(text=f"found: {', '.join(found)}")
                self.btn_scan.configure(state="normal")
                self.scanning = False

            # Update UI on the main thread
            self.after(0, _apply)

        threading.Thread(target=_worker, daemon=True).start()


class ParkingUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Smart Parking")
        self.geometry("1600x900")
        init_db()
        self.verifier = Verifier()

        # Header Banner with project title and team members
        header_frame = ctk.CTkFrame(self, fg_color=("#1f538d", "#14375e"), corner_radius=0)
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header_frame.grid_columnconfigure(0, weight=1)
        
        # Inner container for padding
        header_content = ctk.CTkFrame(header_frame, fg_color="transparent")
        header_content.grid(row=0, column=0, sticky="ew", padx=20, pady=12)
        header_content.grid_columnconfigure(0, weight=1)
        
        # Project Title
        title_label = ctk.CTkLabel(
            header_content,
            text="Smart Parking System With Automatic License Plate Recognition and Face Recognition",
            font=("Arial", 20, "bold"),
            text_color="white"
        )
        title_label.grid(row=0, column=0, pady=(0, 10), sticky="ew")
        
        # Team Members
        members_text = "23110021 Võ Trúc Hồ  |  23110069 Hoàng Đức Tuấn  |  23110057 Trác Văn Ngọc Phúc"
        members_label = ctk.CTkLabel(
            header_content,
            text=members_text,
            font=("Arial", 14),
            text_color="#e0e0e0"
        )
        members_label.grid(row=1, column=0, pady=0, sticky="ew")

        # Tabs: Live cameras + Sessions (DB viewer)
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self.live_tab = self.tabview.add("Live")
        self.sessions_tab = self.tabview.add("Sessions")

        # --- Live tab layout ---
        self.tile_a = CameraTile(self.live_tab, "Camera A", fixed_role="Plate IN")
        self.tile_b = CameraTile(self.live_tab, "Camera B", fixed_role="Face IN")
        self.tile_c = CameraTile(self.live_tab, "Camera C", fixed_role="Plate OUT")
        self.tile_d = CameraTile(self.live_tab, "Camera D", fixed_role="Face OUT")

        # Grid 4 camera tiles with consistent padding and sizing
        self.tile_a.grid(row=0, column=0, padx=(8, 4), pady=(8, 4), sticky="nsew")
        self.tile_b.grid(row=0, column=1, padx=(4, 8), pady=(8, 4), sticky="nsew")
        self.tile_c.grid(row=1, column=0, padx=(8, 4), pady=(4, 8), sticky="nsew")
        self.tile_d.grid(row=1, column=1, padx=(4, 8), pady=(4, 8), sticky="nsew")

        log_label = ctk.CTkLabel(self.live_tab, text="Log", font=("Arial", 14, "bold"))
        log_label.grid(row=0, column=2, padx=8, pady=(8, 0), sticky="ew")
        self.output = ctk.CTkTextbox(self.live_tab, width=400)
        self.output.grid(row=0, column=2, rowspan=2, padx=8, pady=(40, 8), sticky="nsew")

        self.btn_checkin = ctk.CTkButton(self.live_tab, text="Check In", command=self._do_checkin, font=("Arial", 16, "bold"), height=50)
        self.btn_checkout = ctk.CTkButton(self.live_tab, text="Check Out", command=self._do_checkout, font=("Arial", 16, "bold"), height=50)
        self.btn_checkin.grid(row=2, column=0, padx=8, pady=8, sticky="ew")
        self.btn_checkout.grid(row=2, column=1, padx=8, pady=8, sticky="ew")

        self.fee_var = ctk.StringVar(value="0 VND")
        fee_frame = ctk.CTkFrame(self.live_tab)
        fee_frame.grid(row=2, column=2, padx=8, pady=8, sticky="nsew")
        fee_frame.grid_columnconfigure(0, weight=1)
        fee_frame.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(fee_frame, text="Phí (VND)", font=("Arial", 14, "bold")).grid(row=0, column=0, padx=4, pady=(4, 2), sticky="ew")
        self.fee_label = ctk.CTkLabel(
            fee_frame,
            textvariable=self.fee_var,
            font=("Arial", 18, "bold"),
            fg_color="gray20",
            corner_radius=8,
            height=44,
        )
        self.fee_label.grid(row=1, column=0, padx=4, pady=(2, 6), sticky="nsew")

        # Configure grid weights for consistent camera tile sizing
        self.live_tab.grid_columnconfigure(0, weight=1, uniform="live_col")
        self.live_tab.grid_columnconfigure(1, weight=1, uniform="live_col")
        self.live_tab.grid_columnconfigure(2, weight=1)
        self.live_tab.grid_rowconfigure(0, weight=1, uniform="live_row")
        self.live_tab.grid_rowconfigure(1, weight=1, uniform="live_row")

        # --- Sessions tab layout ---
        controls = ctk.CTkFrame(self.sessions_tab)
        controls.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        ctk.CTkLabel(controls, text="Session ID").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        self.db_session_id = ctk.CTkEntry(controls, placeholder_text="session uuid")
        self.db_session_id.grid(row=0, column=1, padx=4, pady=4, sticky="ew")

        ctk.CTkLabel(controls, text="Plate").grid(row=1, column=0, padx=4, pady=4, sticky="w")
        self.db_plate = ctk.CTkEntry(controls, placeholder_text="plate text")
        self.db_plate.grid(row=1, column=1, padx=4, pady=4, sticky="ew")

        ctk.CTkLabel(controls, text="Image path").grid(row=2, column=0, padx=4, pady=4, sticky="w")
        self.db_path = ctk.CTkEntry(controls, placeholder_text="plate image path")
        self.db_path.grid(row=2, column=1, padx=4, pady=4, sticky="ew")

        ctk.CTkLabel(controls, text="Status").grid(row=3, column=0, padx=4, pady=4, sticky="w")
        self.db_status = ctk.CTkOptionMenu(controls, values=["ACTIVE", "EXITED", "CLOSED", "PENDING"])
        self.db_status.set("ACTIVE")
        self.db_status.grid(row=3, column=1, padx=4, pady=4, sticky="ew")

        btn_frame = ctk.CTkFrame(controls)
        btn_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        btn_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkButton(btn_frame, text="Reload", command=self._reload_sessions).grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        ctk.CTkButton(btn_frame, text="Add", command=self._create_session).grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        ctk.CTkButton(btn_frame, text="Update", command=self._update_session).grid(row=0, column=2, padx=4, pady=4, sticky="ew")
        ctk.CTkButton(btn_frame, text="Delete", command=self._delete_session).grid(row=0, column=3, padx=4, pady=4, sticky="ew")

        self.db_text = ctk.CTkTextbox(self.sessions_tab, width=1200, height=700)
        self.db_text.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        self.db_text.configure(state="disabled")

        controls.grid_columnconfigure(1, weight=1)
        self.sessions_tab.grid_rowconfigure(1, weight=1)
        self.sessions_tab.grid_columnconfigure(0, weight=1)

        # Window grid
        self.grid_rowconfigure(0, weight=0)  # Header (fixed height)
        self.grid_rowconfigure(1, weight=1)  # Tabview (expandable)
        self.grid_columnconfigure(0, weight=1)

        # Startup checks and initial data load
        self._check_insightface()
        self._check_paddleocr()
        self._reload_sessions()
        
        # Start real-time fee update (every 1 second)
        self._update_realtime_fee()

    def _check_insightface(self) -> None:
        """Check if InsightFace is properly initialized."""
        try:
            if self.verifier.face._app is None:
                self._log("[STARTUP] WARNING: InsightFace not initialized!")
                self._log("[STARTUP] Install: pip install insightface onnxruntime")
                self._log("[STARTUP] Face verification will NOT work.")
            else:
                self._log("[STARTUP] InsightFace initialized successfully.")
        except Exception as e:
            self._log(f"[STARTUP] Error checking InsightFace: {e}")
    
    def _check_paddleocr(self) -> None:
        """Check if PaddleOCR is properly initialized."""
        try:
            if self.verifier.ocr._paddle is None:
                self._log("[STARTUP] WARNING: PaddleOCR not initialized!")
                self._log("[STARTUP] Install: pip install paddleocr")
                self._log("[STARTUP] License plate recognition will NOT work.")
            else:
                self._log("[STARTUP] PaddleOCR initialized successfully.")
        except Exception as e:
            self._log(f"[STARTUP] Error checking PaddleOCR: {e}")

    def _select_frames_in(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        plate_frame = self.tile_a.get_frame()
        face_frame = self.tile_b.get_frame()
        return plate_frame, face_frame

    def _select_frames_out(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        plate_frame = self.tile_c.get_frame()
        face_frame = self.tile_d.get_frame()
        return plate_frame, face_frame

    def _do_checkin(self) -> None:
        import time
        import cv2
        start_time = time.time()
        
        # Unfreeze Camera B if it was frozen from previous check-in
        if self.tile_b.frozen_frame is not None:
            self.tile_b.unfreeze_frame()
        
        plate_frame, face_frame = self._select_frames_in()
        if plate_frame is None or face_frame is None:
            self._log("[CheckIn] Please assign roles and ensure frames are available.")
            return
        
        # Freeze Camera B (Face IN) frame to preserve check-in image
        self.tile_b.freeze_frame()
        
        # OPTIMIZATION: Resize images before processing to reduce inference time
        # Resize plate image: max 800px width (OCR works well at this size, smaller = faster)
        h, w = plate_frame.shape[:2]
        if w > 800:
            scale = 800.0 / w
            new_w, new_h = int(w * scale), int(h * scale)
            plate_frame = cv2.resize(plate_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Resize face image: max 480px width (face recognition works well at this size)
        h, w = face_frame.shape[:2]
        if w > 480:
            scale = 480.0 / w
            new_w, new_h = int(w * scale), int(h * scale)
            face_frame = cv2.resize(face_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Use optimized verifier (handles parallel processing internally)
        res = self.verifier.handle_entry_frame(plate_frame, face_frame)
        
        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        self._log(f"[CheckIn] session_id={res.session_id[:8]}... plate='{res.plate_text}' status={res.status} | Time: {elapsed_time:.0f}ms")
        
        # Reset fee to 0 after check-in (don't show fee until checkout)
        self.fee_var.set("0 VND")

    def _do_checkout(self) -> None:
        import time
        import cv2
        start_time = time.time()
        
        plate_frame, face_frame = self._select_frames_out()
        if plate_frame is None or face_frame is None:
            self._log("[CheckOut] Please assign roles and ensure frames are available.")
            self.fee_var.set("0 VND")
            return
        
        # OPTIMIZATION: Resize images before processing
        # Resize plate image: max 800px width
        h, w = plate_frame.shape[:2]
        if w > 800:
            scale = 800.0 / w
            new_w, new_h = int(w * scale), int(h * scale)
            plate_frame = cv2.resize(plate_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Resize face image: max 480px width
        h, w = face_frame.shape[:2]
        if w > 480:
            scale = 480.0 / w
            new_w, new_h = int(w * scale), int(h * scale)
            face_frame = cv2.resize(face_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Use optimized verifier (handles parallel processing internally)
        res = self.verifier.handle_exit_frame(plate_frame, face_frame)
        
        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        status_text = "✅ Approved" if res.approved else "⚠️ Not matched"
        self._log(f"[CheckOut] {status_text} | Fee: {res.fee:.0f} VND | Similarity: {res.similarity_score:.3f} | session_id={res.session_id[:8] if res.session_id else 'N/A'}... | Time: {elapsed_time:.0f}ms")
        self.fee_var.set(self._format_fee_vnd(res.fee))
        
        # Unfreeze Camera B (Face IN) after successful check-out to allow next check-in
        if res.approved or res.status in ["COMPLETED", "EXITED"]:
            if self.tile_b.frozen_frame is not None:
                self.tile_b.unfreeze_frame()

    def _reload_sessions(self) -> None:
        from core.db.database import SessionLocal, ParkingSession

        with SessionLocal() as db:
            rows = (
                db.query(ParkingSession)
                .order_by(ParkingSession.time_in.desc())
                .limit(200)
                .all()
            )

        lines = ["session_id | plate | status | time_in | time_out | has_face | sim"]
        for r in rows:
            ti = r.time_in.strftime("%Y-%m-%d %H:%M:%S") if r.time_in else ""
            to = r.time_out.strftime("%Y-%m-%d %H:%M:%S") if r.time_out else ""
            has_face = "yes" if r.get_embedding() is not None else "no"
            sim = f"{r.similarity:.3f}" if r.similarity is not None else "-"
            plate = r.plate_text if r.plate_text else "(empty)"  # Handle None or empty plate_text
            lines.append(f"{r.session_id} | {plate} | {r.status} | {ti} | {to} | {has_face} | {sim}")

        self.db_text.configure(state="normal")
        self.db_text.delete("1.0", "end")
        self.db_text.insert("end", "\n".join(lines))
        self.db_text.configure(state="disabled")

    def _create_session(self) -> None:
        from core.db.database import SessionLocal, ParkingSession

        plate = self.db_plate.get().strip()
        if not plate:
            self._log("[DB] Plate is required to add session")
            return

        path = self.db_path.get().strip() or "manual-entry"
        status = self.db_status.get()

        new_session = ParkingSession(
            session_id=str(uuid.uuid4()),
            plate_text=plate,
            plate_image_path=path,
            time_in=datetime.now(),
            time_out=None,
            status=status,
            similarity=None,
        )
        new_session.set_embedding(None)

        with SessionLocal() as db:
            db.add(new_session)
            db.commit()

        self._log(f"[DB] Added session {new_session.session_id[:8]} for plate {plate}")
        self._reload_sessions()

    def _update_session(self) -> None:
        from core.db.database import SessionLocal, ParkingSession

        sid = self.db_session_id.get().strip()
        if not sid:
            self._log("[DB] Session ID is required for update")
            return

        with SessionLocal() as db:
            sess = db.get(ParkingSession, sid)
            if sess is None:
                self._log(f"[DB] Session {sid} not found")
                return

            updated = False
            plate = self.db_plate.get().strip()
            if plate:
                sess.plate_text = plate
                updated = True

            path = self.db_path.get().strip()
            if path:
                sess.plate_image_path = path
                updated = True

            status = self.db_status.get()
            if status:
                sess.status = status
                # If marking as completed, set time_out when missing
                if status.upper() != "ACTIVE" and sess.time_out is None:
                    sess.time_out = datetime.now()
                updated = True

            if not updated:
                self._log("[DB] Nothing to update")
                return

            db.commit()

        self._log(f"[DB] Updated session {sid}")
        self._reload_sessions()

    def _delete_session(self) -> None:
        from core.db.database import SessionLocal, ParkingSession

        sid = self.db_session_id.get().strip()
        if not sid:
            self._log("[DB] Session ID is required for delete")
            return

        with SessionLocal() as db:
            sess = db.get(ParkingSession, sid)
            if sess is None:
                self._log(f"[DB] Session {sid} not found")
                return
            db.delete(sess)
            db.commit()

        self._log(f"[DB] Deleted session {sid}")
        self._reload_sessions()

    def _log(self, msg: str) -> None:
        self.output.insert("end", msg + "\n")
        self.output.see("end")

    def _format_fee_vnd(self, fee: float) -> str:
        safe_fee = max(0.0, float(fee))
        return f"{safe_fee:,.0f} VND"
    
    def _update_realtime_fee(self) -> None:
        """Keep fee at 0 during active session, only show fee after checkout.
        Fee is set in _do_checkout() after successful checkout.
        """
        from core.db.database import SessionLocal, ParkingSession
        
        try:
            with SessionLocal() as db:
                # Find the most recent active session
                active_session = (
                    db.query(ParkingSession)
                    .filter(ParkingSession.status == "ACTIVE")
                    .order_by(ParkingSession.time_in.desc())
                    .first()
                )
                
                current_value = self.fee_var.get()
                
                if active_session:
                    # During active session, keep fee at 0 (don't show real-time fee)
                    # Only reset to 0 if it's not a checkout fee (checkout fee has comma)
                    if "," not in current_value:
                        self.fee_var.set("0 VND")
                else:
                    # No active session
                    # Only reset to 0 if currently showing 0 or if value doesn't look like a fee
                    # This preserves checkout fee which is set in _do_checkout
                    if current_value == "0 VND" or ("," not in current_value and current_value != "0 VND" and "VND" not in current_value):
                        self.fee_var.set("0 VND")
                    # Otherwise keep the current value (checkout fee)
        except Exception as e:
            # Silently handle errors to avoid spamming logs
            pass
        
        # Schedule next update in 1 second
        self.after(1000, self._update_realtime_fee)


def main() -> None:
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = ParkingUI()
    app.mainloop()


if __name__ == "__main__":
    main()
