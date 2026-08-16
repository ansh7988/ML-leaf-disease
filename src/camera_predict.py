from camera import start_camera
from predict import predict_leaf
from prediction_history import save_prediction
from plot_results import load_history, plot_prediction_counts
import subprocess
import sys


def camera_prediction():

    # Start camera and capture image
    image_path = start_camera()

    # User pressed Q
    if image_path is None:
        print("Camera closed. No image captured.")
        return

    print("\nAnalyzing leaf...")

    # Use existing ML prediction
    result, confidence = predict_leaf(image_path)

    print("\nCamera Prediction")
    print("----------------------")
    print(f"Image: {image_path}")
    print(f"Result: {result}")
    print(f"Confidence: {confidence * 100:.2f}%")

    # Save prediction
    save_prediction(result, confidence)

    print("\nPrediction saved.")
    print("Updating graphs...")

    # Update result graph
    history = load_history()
    plot_prediction_counts(history)

    # Open confidence graph
    subprocess.run(
        [sys.executable, "src/plot_confidence.py"]
    )


if __name__ == "__main__":
    camera_prediction()