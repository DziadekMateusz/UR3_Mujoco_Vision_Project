import cv2

from inference.verify_object import verify_frame, draw_detections

REQUIRED_FRAMES = 15

def run_detection():

    """Capture frames until accepted for (REQUIRED_FRAMES)
    consecutive frames, then return detection. Returns None if
    the camera feed ends (or ESC is pressed)."""

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    verified_counter = 0

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            detections = verify_frame(frame)
            draw_detections(frame, detections)

            accepted = [d for d in detections if d["accepted"]]
            accepted_detection = accepted[-1] if accepted else None

            cv2.imshow("Two Stage Detection", frame)

            key = cv2.waitKey(1)
            if key == 27:
                break

            if accepted_detection is not None:
                verified_counter += 1
                print(f"VERIFIED {verified_counter}/{REQUIRED_FRAMES}")
            else:
                verified_counter = 0

            if verified_counter >= REQUIRED_FRAMES:
                print("RETURNING DETECTION")
                return accepted_detection

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return None

if __name__ == "__main__":
    result = run_detection()
    print(result)
