import sys
import tensorflow as tf
import numpy as np

from src.prediction_history import save_prediction


MODEL_PATH = "model/leaf_health_model.keras"

IMG_SIZE = (224, 224)

# Model classes
CLASS_NAMES = [
    "Healthy",
    "Insect Pest",
    "Leaf Blight",
    "Leaf Spot",
    "Nutrient Stress",
    "Powdery Mildew",
    "Rust"
]

# Below this confidence, don't claim an exact disease
CONFIDENCE_THRESHOLD = 0.60


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
    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    # Get highest probability class
    predicted_index = np.argmax(predictions)

    confidence = float(predictions[predicted_index])

    predicted_class = CLASS_NAMES[predicted_index]


    # ---------------------------------
    # UNCERTAIN PREDICTION
    # ---------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        if predicted_class == "Healthy":

            result = "Healthy, but prediction is uncertain"

        else:

            result = "Diseased, but exact disease could not be identified"

    else:

        result = predicted_class


    # Save prediction
    save_prediction(result, confidence)


    return (
        result,
        confidence,
        predictions
    )


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)


    image_path = sys.argv[1]

    result, confidence, predictions = predict_leaf(image_path)


    print("\nPrediction")
    print("----------------")

    print(f"Result: {result}")
    print(f"Confidence: {confidence * 100:.2f}%")


    print("\nAll Class Probabilities")
    print("----------------")

    for class_name, probability in zip(
        CLASS_NAMES,
        predictions
    ):

        print(
            f"{class_name}: {probability * 100:.2f}%"
        )