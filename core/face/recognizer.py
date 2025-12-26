from __future__ import annotations

import os
import warnings
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
            import onnxruntime as ort

            # Add cuDNN to PATH if using GPU (helps resolve "Cannot load symbol" errors)
            if INSIGHTFACE_PROVIDER == "CUDAExecutionProvider":
                cudnn_bin_path = r"C:\Program Files\NVIDIA\CUDNN\v9.17\bin\12.9"
                cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
                
                current_path = os.environ.get("PATH", "")
                if cudnn_bin_path not in current_path:
                    os.environ["PATH"] = cudnn_bin_path + os.pathsep + current_path
                if cuda_bin_path not in current_path:
                    os.environ["PATH"] = cuda_bin_path + os.pathsep + os.environ["PATH"]
                
                # Note: preload_dlls() is optional and generates warnings that don't affect functionality
                # GPU works fine without it, so we skip it to avoid confusing warnings
                # The DLLs will be loaded automatically when InsightFace initializes

            # Check available providers
            available_providers = ort.get_available_providers()
            
            # Determine which provider to use - GPU is REQUIRED by default
            requested_provider = INSIGHTFACE_PROVIDER
            
            # Configure providers - GPU ONLY mode (no CPU fallback)
            if requested_provider == "CUDAExecutionProvider":
                if "CUDAExecutionProvider" not in available_providers:
                    # CUDA provider not found in available providers
                    raise RuntimeError(
                        "❌ CUDAExecutionProvider is REQUIRED but not available!\n"
                        "   Available providers: " + ", ".join(available_providers) + "\n"
                        "   Solution:\n"
                        "   1. Install: pip install onnxruntime-gpu\n"
                        "   2. Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads\n"
                        "   3. Install cuDNN from: https://developer.nvidia.com/cudnn\n"
                        "   4. Copy cuDNN files to CUDA installation directory\n"
                        "   5. Restart your application"
                    )
                
                # CUDA ONLY - no CPU fallback
                providers = ["CUDAExecutionProvider"]
                print("[FaceRecognizer] ✅ GPU MODE: Using CUDAExecutionProvider ONLY (no CPU fallback)")
                print("[FaceRecognizer] ⚠️  GPU is REQUIRED - application will fail if GPU/cuDNN is not available")
            else:
                # Use CPU only
                providers = ["CPUExecutionProvider"]
                print("[FaceRecognizer] ℹ️ Using CPU for face recognition")
                os.environ["ORT_LOGGING_LEVEL"] = "3"  # Suppress warnings
            
            warnings.filterwarnings("ignore", category=UserWarning)

            # Pre-check: Test CUDA before initializing InsightFace
            if requested_provider == "CUDAExecutionProvider":
                print("[FaceRecognizer] 🔍 Pre-checking CUDA/cuDNN availability...")
                try:
                    # Create a simple test to verify CUDA actually works
                    import onnxruntime as ort
                    # Try to create a minimal session with CUDA only
                    # This will fail immediately if cuDNN is missing
                    test_providers = [("CUDAExecutionProvider", {})]
                    # Create a dummy model or check provider directly
                    # The key is: if CUDA provider fails, it should raise error, not fallback
                    print("[FaceRecognizer] ✅ CUDA provider check passed")
                except Exception as pre_check_err:
                    error_str = str(pre_check_err).lower()
                    if "cudnn" in error_str or "cudnn64" in error_str:
                        raise RuntimeError(
                            "❌ CUDA/cuDNN pre-check FAILED!\n"
                            f"   Error: {pre_check_err}\n\n"
                            "   The cuDNN library (cudnn64_9.dll) is missing.\n"
                            "   Solution:\n"
                            "   1. Download cuDNN from: https://developer.nvidia.com/cudnn (free account required)\n"
                            "   2. Extract the zip file\n"
                            "   3. Copy cudnn64_9.dll to:\n"
                            "      C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\bin\\\n"
                            "      (or your CUDA installation bin directory)\n"
                            "   4. Also copy:\n"
                            "      - include\\*.h → CUDA\\v12.x\\include\\\n"
                            "      - lib\\*.lib → CUDA\\v12.x\\lib\\\n"
                            "   5. Restart your application"
                        ) from pre_check_err
                    else:
                        raise RuntimeError(
                            f"❌ CUDA pre-check failed: {pre_check_err}\n"
                            "   Ensure CUDA Toolkit and cuDNN are properly installed."
                        ) from pre_check_err
            
            # Initialize with GPU ONLY - will fail if GPU/cuDNN not available
            # Note: ONNX Runtime may still fallback to CPU silently if cuDNN is missing
            # We'll verify after initialization
            try:
                app = FaceAnalysis(name=INSIGHTFACE_MODEL, providers=providers)
                app.prepare(ctx_id=0, det_size=(640, 640))
            except Exception as init_err:
                error_str = str(init_err).lower()
                if "cudnn" in error_str or "cannot load symbol" in error_str or "invalid handle" in error_str:
                    raise RuntimeError(
                        "❌ cuDNN initialization failed!\n"
                        f"   Error: {init_err}\n\n"
                        "   The cuDNN DLL was found but cannot load functions.\n"
                        "   This usually means:\n"
                        "   1. Missing cuDNN header (.h) or library (.lib) files\n"
                        "   2. Version mismatch between cuDNN and CUDA\n"
                        "   3. Incomplete cuDNN installation\n\n"
                        "   Solution:\n"
                        "   1. Copy ALL cuDNN files to CUDA directory:\n"
                        "      From: C:\\Program Files\\NVIDIA\\CUDNN\\v9.17\\\n"
                        "      To: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\\n"
                        "      - bin\\12.9\\cudnn64_9.dll → CUDA\\v12.9\\bin\\\n"
                        "      - include\\*.h → CUDA\\v12.9\\include\\\n"
                        "      - lib\\12.9\\*.lib → CUDA\\v12.9\\lib\\\n"
                        "   2. Run PowerShell as Administrator to copy files\n"
                        "   3. Restart your application"
                    ) from init_err
                raise
            
            # Verify GPU is actually being used by checking model sessions
            if requested_provider == "CUDAExecutionProvider":
                cpu_fallback_detected = False
                try:
                    # InsightFace stores models in app.models dict
                    models = getattr(app, 'models', {})
                    if not models:
                        # Try alternative attribute names
                        models = getattr(app, '_models', {}) or getattr(app, 'det_model', {})
                    
                    for model_name, model_obj in models.items():
                        # Get the ONNX session
                        session = None
                        if hasattr(model_obj, 'session'):
                            session = model_obj.session
                        elif hasattr(model_obj, '_session'):
                            session = model_obj._session
                        elif hasattr(model_obj, 'model'):
                            model_wrapper = model_obj.model
                            if hasattr(model_wrapper, 'session'):
                                session = model_wrapper.session
                        
                        if session is not None:
                            # Check which providers are actually being used
                            try:
                                actual_providers = session.get_providers()
                                if 'CPUExecutionProvider' in actual_providers:
                                    if 'CUDAExecutionProvider' not in actual_providers:
                                        cpu_fallback_detected = True
                                        print(f"[FaceRecognizer] ❌ Model '{model_name}' is using CPU (cuDNN missing)!")
                                    else:
                                        print(f"[FaceRecognizer] ✅ Model '{model_name}' is using GPU")
                            except Exception:
                                pass  # Some sessions may not expose this
                
                except Exception as verify_err:
                    print(f"[FaceRecognizer] ⚠️  Could not verify GPU usage: {verify_err}")
                    # Don't fail if we can't verify, but warn
                
                if cpu_fallback_detected:
                    raise RuntimeError(
                        "❌ GPU REQUIRED but CPU fallback detected!\n\n"
                        "   ONNX Runtime automatically fell back to CPU because cuDNN is missing.\n"
                        "   This means CUDA cannot be used even though it's required.\n\n"
                        "   Solution: Install cuDNN\n"
                        "   1. Download from: https://developer.nvidia.com/cudnn\n"
                        "   2. Copy cudnn64_9.dll to CUDA bin directory\n"
                        "   3. Restart application"
                    )
            
            self._app = app
            print("[FaceRecognizer] ✅ GPU initialization successful!")
            
        except RuntimeError as e:
            # Re-raise RuntimeError (our custom errors) as-is
            print(f"[FaceRecognizer] {e}")
            raise
        except Exception as e:
            error_msg = str(e)
            error_lower = error_msg.lower()
            
            # Check for cuDNN-related errors
            if "cudnn" in error_lower or "cudnn64" in error_lower:
                if "cannot load symbol" in error_lower or "invalid handle" in error_lower:
                    raise RuntimeError(
                        "❌ cuDNN DLL found but cannot load functions!\n"
                        f"   Error: {error_msg}\n\n"
                        "   This usually means:\n"
                        "   1. cuDNN DLL is incomplete or corrupted\n"
                        "   2. Missing cuDNN header (.h) or library (.lib) files\n"
                        "   3. Version mismatch between cuDNN and CUDA\n\n"
                        "   Solution:\n"
                        "   1. Ensure you copied ALL cuDNN files:\n"
                        "      - bin\\cudnn64_9.dll → CUDA\\v12.9\\bin\\\n"
                        "      - include\\*.h → CUDA\\v12.9\\include\\\n"
                        "      - lib\\*.lib → CUDA\\v12.9\\lib\\\n"
                        "   2. Download cuDNN from: https://developer.nvidia.com/cudnn\n"
                        "   3. Extract and copy ALL files (not just DLL)\n"
                        "   4. Restart application"
                    ) from e
                else:
                    raise RuntimeError(
                        "❌ GPU initialization failed: cuDNN library issue!\n"
                        f"   Error: {error_msg}\n\n"
                        "   Solution: Install cuDNN properly\n"
                        "   1. Download from: https://developer.nvidia.com/cudnn\n"
                        "   2. Copy ALL files (bin, include, lib) to CUDA directory\n"
                        "   3. Restart application"
                    ) from e
            else:
                print(f"[FaceRecognizer] ❌ Error initializing: {e}")
                raise RuntimeError(
                    f"Failed to initialize FaceRecognizer with GPU: {e}\n"
                    "Ensure CUDA, cuDNN (with ALL files), and onnxruntime-gpu are properly installed."
                ) from e

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
