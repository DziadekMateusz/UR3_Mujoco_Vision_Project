from inference.realtime_detection import run_detection
from simulation.robot_controller import RobotController


def main():
    print("\n========== SYSTEM START ==========\n")

    detection = run_detection()

    if detection is None:
        print("No verified object detected.")
        return

    detected_class = detection["cnn_class"]
    print(f"\nVERIFIED OBJECT: {detected_class}")

    controller = RobotController()
    controller.pick_and_place(detected_class)

    print("\nTASK COMPLETED")


if __name__ == "__main__":
    main()
