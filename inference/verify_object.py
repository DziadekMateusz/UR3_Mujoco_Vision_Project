import os
import cv2
import numpy as np
import tensorflow as tf

from ultralytics import YOLO

# Models paths

ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

YOLO_MODEL = os.path.join(
    ROOT, "models", "yolo", "earphones_mouse_ram", "weights", "best.pt"
)

CNN_MODEL = os.path.join(
    ROOT, "models", "cnn", "tm_cnn.keras"
)

# Config

CLASS_NAMES = ["earphones", "mouse", "ram"]

CNN_THRESHOLD = 0.70
YOLO_THRESHOLD = 0.50

# Loading models

print("\nLOADING YOLO...")
yolo = YOLO(YOLO_MODEL)

print("LOADING CNN...")
cnn = tf.keras.models.load_model(CNN_MODEL)

print("MODELS READY\n")

# Image preprocessing

def preprocess_crop(crop):
    crop = cv2.resize(crop, (224, 224))
    crop = crop.astype(np.float32)
    crop /= 255.0
    return np.expand_dims(crop, axis=0)

# Verify detection

def verify_frame(frame):

    # Run YOLO detection on a frame, then verify every box with the CNN

    height, width = frame.shape[:2]

    results = yolo(frame, conf=YOLO_THRESHOLD)

    candidates = []   # (cls, conf, x1, y1, x2, y2), same order as crop_batch
    crop_batch = []   # preprocessed crops ready to feed the CNN

    for result in results:
        for box in result.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            try:
                crop_batch.append(preprocess_crop(crop))
            except Exception as e:
                print("CNN PREPROCESS ERROR:", e)
                continue

            candidates.append((cls, conf, x1, y1, x2, y2))

    detections = []

    if not crop_batch:
        return detections

    try:
        preds = cnn.predict(
            np.concatenate(crop_batch, axis=0),
            verbose=0
        )
    except Exception as e:
        print("CNN ERROR:", e)
        return detections

    for (cls, conf, x1, y1, x2, y2), pred in zip(candidates, preds):

        cnn_class = int(np.argmax(pred))
        cnn_conf = float(np.max(pred))

        accepted = (
            cls < len(CLASS_NAMES)
            and cnn_class == cls
            and cnn_conf >= CNN_THRESHOLD
        )

        print(
            "YOLO:", CLASS_NAMES[cls], conf,
            "| CNN:", CLASS_NAMES[cnn_class], cnn_conf,
            "| ACCEPTED:", accepted
        )

        detections.append({
            "accepted": accepted,
            "bbox": (x1, y1, x2, y2),
            "yolo_class": CLASS_NAMES[cls],
            "yolo_conf": conf,
            "cnn_class": CLASS_NAMES[cnn_class],
            "cnn_conf": cnn_conf
        })

    return detections

# Drawing detections

def draw_detections(frame, detections):
    """Draws bounding boxes + labels for a list of verify_frame() results."""

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = (0, 255, 0) if d["accepted"] else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{d['yolo_class']} Y:{d['yolo_conf']:.2f} C:{d['cnn_conf']:.2f}"

        cv2.putText(
            frame, text, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    return frame

# Test using camera

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            detections = verify_frame(frame)
            draw_detections(frame, detections)

            cv2.imshow("Verification", frame)

            key = cv2.waitKey(1)
            if key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
