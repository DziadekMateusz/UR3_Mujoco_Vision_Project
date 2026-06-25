import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = ROOT / "models" / "yolo" / "yolov8n.pt"
DATA_YAML = ROOT / "datasets" / "detection" / "data.yaml"
OUTPUT_DIR = ROOT / "models" / "yolo"

EPOCHS = 100
IMG_SIZE = 640
BATCH = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(str(MODEL_NAME))

cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)
print("GPU:", torch.cuda.get_device_name(0) if cuda_available else "none (training on CPU)")

model.train(
    data=str(DATA_YAML.resolve()),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    workers=0,
    device=0 if cuda_available else "cpu",
    project=str(OUTPUT_DIR.resolve()),
    name="earphones_mouse_ram"
)

print("\nTRAINING FINISHED")
