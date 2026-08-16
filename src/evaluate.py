import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import get_data_generators


MODEL_PATH = "model/leaf_health_model.keras"


# Load validation data
_, validation_data = get_data_generators()

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Evaluate
loss, accuracy = model.evaluate(validation_data)

print("\nEvaluation Results")
print("------------------")
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")


# Predictions
predictions = model.predict(validation_data)

predicted_classes = (predictions > 0.5).astype(int).flatten()
true_classes = validation_data.classes

class_names = list(validation_data.class_indices.keys())

print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        true_classes,
        predicted_classes,
        labels=[0, 1],
        target_names=["diseased", "healthy"],
        zero_division=0
    )
)

print("\nConfusion Matrix")
print("----------------")
print(confusion_matrix(true_classes, predicted_classes,labels=[0, 1]))