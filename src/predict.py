import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from src.prediction_history import save_prediction


MODEL_PATH = "model/leaf_health_model.keras"

IMG_SIZE = (224, 224)


# Load model
model = tf.keras.models.load_model(MODEL_PATH)

def predict_leaf(image_path):

    # Load image
    img = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    # Convert image to array
    img_array = tf.keras.utils.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # MobileNetV2 preprocessing
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_array
    )

    # Model prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )[0][0]

    # Probabilities
    healthy_probability = float(prediction)
    diseased_probability = float(1 - prediction)

    # Final prediction
    if prediction >= 0.5:

        result = "Healthy"
        confidence = healthy_probability

    else:

        result = "Diseased"
        confidence = diseased_probability

    save_prediction(result, confidence)

    return (
        result,
        confidence,
        healthy_probability,
        diseased_probability
    )

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    result, confidence = predict_leaf(image_path)

    print("\nPrediction")
    print("----------------")
    print(f"Result: {result}")
    print(f"Confidence: {confidence * 100:.2f}%")