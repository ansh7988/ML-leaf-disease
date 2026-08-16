import cv2
import os
from datetime import datetime


SAVE_DIR = "data/captured"


def start_camera():

    # Create folder if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not access the camera.")
        return None

    print("Camera started.")
    print("Press C to capture a leaf.")
    print("Press Q to quit.")

    captured_path = None

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Error: Could not read camera frame.")
            break

        # Display instructions on camera window
        cv2.putText(
            frame,
            "Press C = Capture | Q = Quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Leaf Health Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        # Capture
        if key == ord("c"):

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"leaf_{timestamp}.jpg"

            captured_path = os.path.join(
                SAVE_DIR,
                filename
            )

            cv2.imwrite(captured_path, frame)

            print(f"\nLeaf image captured:")
            print(captured_path)

            break

        # Quit
        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    return captured_path


if __name__ == "__main__":

    start_camera()