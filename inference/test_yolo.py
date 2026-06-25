import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
from pathlib import Path

def main():

    ROOT = Path(__file__).resolve().parent.parent

    MODEL_PATH = (
        ROOT
        / "models"
        / "yolo"
        / "earphones_mouse_ram"
        / "weights"
        / "best.pt"
    )

    DATASET_YAML = (
        ROOT
        / "datasets"
        / "detection"
        / "data.yaml"
    )

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATASET_YAML),
        split="val",
        imgsz=640
    )

    print("\n========== YOLO VALIDATION ==========")

    print(f"mAP50      : {metrics.box.map50:.4f}")
    print(f"mAP50-95   : {metrics.box.map:.4f}")
    print(f"Precision  : {metrics.box.mp:.4f}")
    print(f"Recall     : {metrics.box.mr:.4f}")

    print("=====================================\n")

if __name__ == "__main__":
    main()