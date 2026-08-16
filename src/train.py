import os
import tensorflow as tf

from data_loader import get_data_generators
from model import build_model


# -----------------------------
# 1. Load data
# -----------------------------

train_data, validation_data = get_data_generators()

print("\nClass mapping:")
print(train_data.class_indices)


# -----------------------------
# 2. Build model
# -----------------------------

model = build_model()

model.summary()


# -----------------------------
# 3. Compile model
# -----------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------
# 4. Train model
# -----------------------------

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=10
)


# -----------------------------
# 5. Save model
# -----------------------------

os.makedirs("model", exist_ok=True)

model.save("model/leaf_health_model.keras")

print("\nTraining completed!")
print("Model saved at:")
print("model/leaf_health_model.keras")