import tensorflow as tf
import numpy as np
import cv2


# --------------------------------------------------
# MODEL
# --------------------------------------------------

MODEL_PATH = "model/leaf_health_model.keras"

IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)


# MobileNetV2 inside our model
base_model = model.get_layer("mobilenetv2_1.00_224")

# Last convolutional layer
last_conv_layer = base_model.get_layer("block_16_project")


# --------------------------------------------------
# GRAD-CAM
# --------------------------------------------------

def make_gradcam(image_path):

    # Load image
    img = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    img_array = tf.keras.utils.img_to_array(img)

    # Keep original image for overlay
    original = img_array.astype(np.uint8)

    # Prepare image for model
    input_array = np.expand_dims(
        img_array,
        axis=0
    )

    input_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        input_array
    )

    # --------------------------------------------------
    # GRADIENT MODEL
    # --------------------------------------------------

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    # Get feature maps and model output
    with tf.GradientTape() as tape:

        conv_outputs, base_output = grad_model(
            input_array
        )

        # Continue through the outer model
        x = model.layers[1](base_output)
        x = model.layers[2](x)
        x = model.layers[3](x)
        predictions = model.layers[4](x)

        # Class score
        class_score = predictions[:, 0]

    # Gradients
    grads = tape.gradient(
        class_score,
        conv_outputs
    )

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    # Feature maps
    conv_outputs = conv_outputs[0]

    pooled_grads = pooled_grads[0]

    # Weight feature maps
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # ReLU
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    max_value = tf.reduce_max(heatmap)

    heatmap = heatmap / (
        max_value + tf.keras.backend.epsilon()
    )

    heatmap = heatmap.numpy()

    # --------------------------------------------------
    # CREATE HEATMAP
    # --------------------------------------------------

    heatmap = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0])
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # --------------------------------------------------
    # OVERLAY
    # --------------------------------------------------

    original_bgr = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )

    overlay = cv2.addWeighted(
        original_bgr,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    # Convert back to RGB
    overlay = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )

    return overlay