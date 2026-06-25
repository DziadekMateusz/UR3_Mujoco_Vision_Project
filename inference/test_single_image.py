import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = ROOT / "test_images" / "earphones_test.jpg"

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise FileNotFoundError(
        f"Cannot load image: {IMAGE_PATH}"
    )

from verify_object import verify_frame

detections = verify_frame(image)

print("\nRESULTS:\n")

for d in detections:

    print(
        f"YOLO={d['yolo_class']} "
        f"{d['yolo_conf']:.2f} | "
        f"CNN={d['cnn_class']} "
        f"{d['cnn_conf']:.2f} | "
        f"ACCEPTED={d['accepted']}"
    )