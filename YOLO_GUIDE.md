# YOLO + OCR Pipeline cho Biển Số Việt Nam

## 🎯 Chiến lược

```
Ảnh → YOLO Detection (locate plate) → Crop ROI → PaddleOCR → Text
```

Nếu YOLO chưa có: OCR trên toàn ảnh (fallback)

---

## 📦 Chuẩn bị Dataset

### Định dạng YOLO:
```
dataset/
├── images/
│   ├── train/  (100-500 ảnh)
│   ├── val/    (20-100 ảnh)
│   └── test/   (10-50 ảnh)
└── labels/     (YOLO format .txt)
    ├── train/
    ├── val/
    └── test/
```

### File nhãn (.txt):
- 1 file .txt per ảnh
- Format: `<class_id> <x_center> <y_center> <width> <height>` (normalized 0-1)
- Ví dụ: `0 0.5 0.5 0.3 0.2` (class=plate, centered)

### Tool annotate:
- [Roboflow](https://roboflow.com/) - free, convert sang YOLO
- [LabelImg](https://github.com/heartexer/labelImg) - desktop tool
- [CVAT](https://cvat.org/) - web-based

---

## 🚀 Train YOLO

### Bước 1: Chuẩn bị `data.yaml`
```yaml
path: /full/path/to/dataset
train: images/train
val: images/val
test: images/test
nc: 1
names: ['plate']
```

### Bước 2: Train
```bash
# CPU (chậm nhưng không cần GPU)
python train_plate_detector.py --data data.yaml --epochs 100 --imgsz 640 --device cpu

# CUDA (nếu có GPU)
python train_plate_detector.py --data data.yaml --epochs 100 --imgsz 640 --device 0
```

Kết quả:
```
models/plate_detector/weights/
├── best.pt     ← Dùng cái này
└── last.pt
```

---

## 💾 Sử dụng Model

### Cách 1: Env variable
```bash
# PowerShell
$env:YOLO_PLATE_MODEL = "models/plate_detector/weights/best.pt"
python -m uvicorn core.api.main:app --reload

# CMD
set YOLO_PLATE_MODEL=models\plate_detector\weights\best.pt
python -m uvicorn core.api.main:app --reload
```

### Cách 2: Code
```python
from core.decision.verifier import Verifier

verifier = Verifier()  # Tự load từ YOLO_PLATE_MODEL env var

# Hoặc truyền trực tiếp:
from core.anpr.detector import LicensePlateDetector
from core.anpr.ocr import PlateOCR

ocr = PlateOCR()
detector = LicensePlateDetector(ocr, model_path="models/plate_detector/weights/best.pt")
```

---

## 🧪 Test Pipeline

### Test OCR only (hiện tại):
```bash
python test_ocr_quick.py
```

### Test với API:
```bash
# Terminal 1: Start server
python -m uvicorn core.api.main:app --reload

# Terminal 2: Send request
curl -X POST http://localhost:8000/entry \
  -F "plate_image=@plate.jpg" \
  -F "face_image=@face.jpg"
```

---

## 📊 Tuning

### Nếu YOLO miss (không detect):
- Tăng dataset (500+ ảnh)
- Tăng epochs (150-200)
- Adjust conf threshold trong detector.py

### Nếu OCR sai:
- Cải thiện quality ảnh crop (YOLO bbox)
- Điều chỉnh `PLATE_REGEX` trong config.py
- Debug: `DEBUG_OCR=1 python ...`

### Nếu tối ưu tốc độ:
- Dùng YOLOv8n (nano) - nhẹ
- Dùng imgsz=416 (nhỏ hơn 640)
- Chạy FP16 thay FP32 (CUDA only)

---

## 📚 Tài liệu

- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [Dataset format](https://roboflow.com/formats/yolov8)

---

## ✅ Checklist

- [ ] Collect 200+ plate images (Vietnam)
- [ ] Annotate with YOLO format
- [ ] Create data.yaml
- [ ] Run training
- [ ] Eval metrics (mAP, F1)
- [ ] Export best.pt
- [ ] Set YOLO_PLATE_MODEL env
- [ ] Test entry/exit flow
