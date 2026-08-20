import tensorflow as tf
import numpy as np
import cv2


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "model/leaf_health_model.keras"
IMG_SIZE = (224, 224)
NUM_CLASSES = 7


# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(MODEL_PATH)

# Force the outer Sequential model to be built/called.
# This prevents the "Sequential has never been called"
# problem when accessing its graph.
dummy_input = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
_ = model(dummy_input, training=False)


# ============================================================
# GET MOBILENETV2
# ============================================================

base_model = model.get_layer("mobilenetv2_1.00_224")

# Force MobileNetV2 itself to be called as well.
_ = base_model(dummy_input, training=False)


# ============================================================
# LAST CONVOLUTIONAL LAYER
# ============================================================

last_conv_layer = base_model.get_layer("Conv_1")


# ============================================================
# CREATE GRAD-CAM MODEL
# ============================================================

# IMPORTANT:
# We use base_model.inputs rather than model.input.
#
# The output contains:
#   1. Last convolutional feature maps
#   2. MobileNetV2 final output
#
# This avoids the problematic outer Sequential input graph.

grad_model = tf.keras.Model(
    inputs=base_model.inputs,
    outputs=[
        last_conv_layer.output,
        base_model.output
    ]
)


# ============================================================
# CLASSIFICATION HEAD
# ============================================================

# Your outer model is:
#
# MobileNetV2
#      ↓
# GlobalAveragePooling2D
#      ↓
# Dense(128)
#      ↓
# Dropout
#      ↓
# Dense(7)
#
# We take everything AFTER MobileNetV2.

global_pool = model.layers[1]
dense_128 = model.layers[2]
dropout = model.layers[3]
classifier = model.layers[4]


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def make_gradcam(image_path):

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    img = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    img_array = tf.keras.utils.img_to_array(img)

    # Original image for final overlay
    original = img_array.astype(np.uint8)

    # --------------------------------------------------------
    # PREPROCESS IMAGE
    # --------------------------------------------------------

    input_array = np.expand_dims(
        img_array,
        axis=0
    ).astype(np.float32)

    input_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        input_array
    )

    # --------------------------------------------------------
    # GRADIENT TAPE
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        # Get convolutional feature maps
        conv_outputs, base_output = grad_model(
            input_array,
            training=False
        )

        # Pass MobileNetV2 output through classifier
        x = global_pool(base_output)
        x = dense_128(x)
        x = dropout(x, training=False)
        predictions = classifier(x)

        # ----------------------------------------------------
        # IMPORTANT:
        # Use the CLASS ACTUALLY PREDICTED.
        # ----------------------------------------------------

        predicted_class = tf.argmax(
            predictions[0]
        )

        class_score = predictions[
            :, predicted_class
        ]

    # --------------------------------------------------------
    # GRADIENTS
    # --------------------------------------------------------

    grads = tape.gradient(
        class_score,
        conv_outputs
    )

    # Safety check
    if grads is None:
        raise RuntimeError(
            "Gradients are None. "
            "Grad-CAM could not connect the target layer "
            "to the prediction."
        )

    # --------------------------------------------------------
    # GLOBAL AVERAGE POOLING
    # --------------------------------------------------------

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]

    # --------------------------------------------------------
    # WEIGHT FEATURE MAPS
    # --------------------------------------------------------

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # --------------------------------------------------------
    # RELU
    # --------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    max_value = tf.reduce_max(heatmap)

    if float(max_value) > 0:
        heatmap = heatmap / max_value
    else:
        heatmap = tf.zeros_like(heatmap)

    heatmap = heatmap.numpy()

    # --------------------------------------------------------
    # RESIZE HEATMAP
    # --------------------------------------------------------

    heatmap = cv2.resize(
        heatmap,
        (
            original.shape[1],
            original.shape[0]
        )
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    # --------------------------------------------------------
    # COLOR MAP
    # --------------------------------------------------------

    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # --------------------------------------------------------
    # OVERLAY
    # --------------------------------------------------------

    original_bgr = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )

    overlay = cv2.addWeighted(
        original_bgr,
        0.60,
        heatmap_color,
        0.40,
        0
    )

    # Convert back to RGB
    overlay = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )

    return overlay