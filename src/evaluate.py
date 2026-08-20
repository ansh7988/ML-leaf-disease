import tensorflow as tf
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from data_loader import get_data_generators


MODEL_PATH = "model/leaf_health_model.keras"

CLASS_NAMES = [
    "Healthy",
    "Insect Pest",
    "Leaf Blight",
    "Leaf Spot",
    "Nutrient Stress",
    "Powdery Mildew",
    "Rust"
]


# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load validation data
_, validation_data = get_data_generators()

# Make predictions
predictions = model.predict(
    validation_data,
    verbose=1
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

true_classes = validation_data.classes


# Accuracy
accuracy = accuracy_score(
    true_classes,
    predicted_classes
)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# Classification report
print("\nClassification Report")
print("------------------------------")

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# Confusion matrix
cm = confusion_matrix(
    true_classes,
    predicted_classes
)

print("\nConfusion Matrix")
print("------------------------------")

print(cm)