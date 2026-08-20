import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


NUM_CLASSES = 7


def build_model():

    # Load pretrained MobileNetV2
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze pretrained layers
    base_model.trainable = False

    model = models.Sequential([

        # Pretrained CNN
        base_model,

        # Convert feature maps into a vector
        layers.GlobalAveragePooling2D(),

        # Classification features
        layers.Dense(128, activation="relu"),

        # Prevent overfitting
        layers.Dropout(0.5),

        # 7-class classification
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model