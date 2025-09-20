import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
import numpy as np
import cv2

IMG_SIZE = (128, 128)

# ----------------------------
# Build model
# ----------------------------
def build_model(input_shape=(128,128,3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Conv2D(128, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

@st.cache_resource
def load_trained_model():
    model = build_model((IMG_SIZE[0], IMG_SIZE[1], 3))
    model.load_weights("model_main.keras")  # adjust path
    _ = model(tf.random.normal((1, IMG_SIZE[0], IMG_SIZE[1],3)))  # dummy call
    return model

model = load_trained_model()

# ----------------------------
# Grad-CAM
# ----------------------------
def make_gradcam_heatmap(img_array, model):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            last_conv_layer = layer.name
            break

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:,0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ----------------------------
# Streamlit App
# ----------------------------
st.title("🔎 Deepfake Image Detection with Grad-CAM")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = np.expand_dims(np.array(img.resize(IMG_SIZE))/255.0, axis=0)
    pred = model.predict(img_array)[0][0]
    label = "Real ✅" if pred>0.5 else "Fake ❌"
    st.subheader(f"Prediction: {label}")

    if label.startswith("Fake"):
        heatmap = make_gradcam_heatmap(img_array, model)
        heatmap_resized = cv2.resize(heatmap, (img.width, img.height))
        heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlayed_img = np.uint8(0.5*np.array(img) + 0.5*heatmap_colored)
        st.image(overlayed_img, caption="Grad-CAM Overlay", use_container_width=True)
