# ----------------------------
# app.py - Deepfake Detection + Grad-CAM (Cloud-friendly)
# ----------------------------
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import cv2
import os
import gdown

# ----------------------------
# Constants
# ----------------------------
IMG_SIZE = (128, 128)
MODEL_PATH = "deepfake_detector.keras"
GDRIVE_ID = "1LatU_PLtaXDuWUWlTOCHMFJ7eTOJGRQP"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"

# ----------------------------
# Download model if not exists
# ----------------------------
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# ----------------------------
# Load model safely
# ----------------------------
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ----------------------------
# Grad-CAM
# ----------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_2"):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:,0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    return heatmap.numpy()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🖼️ Deepfake Detection + Grad-CAM")
st.write("Upload images to detect Real vs Fake content.")

uploaded_files = st.file_uploader(
    "Choose images", type=['jpg','jpeg','png'], accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        img_array = np.expand_dims(np.array(img.resize(IMG_SIZE))/255.0, axis=0)

        # Prediction
        pred = model.predict(img_array)
        label = "Real" if pred[0][0] > 0.5 else "Fake"
        st.write(f"**{file.name} → Prediction:** {label}")

        # Grad-CAM overlay for Fake
        if label == "Fake":
            last_conv_layer_name = None
            # Find the last Conv2D layer
            for layer in reversed(model.layers):
                if isinstance(layer, layers.Conv2D):
                    last_conv_layer_name = layer.name
                    break

            if last_conv_layer_name:
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
                heatmap_resized = cv2.resize(heatmap, (img.width, img.height))
                heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap_resized), cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                overlayed_img = np.array(img)*0.5 + heatmap_colored*0.5
                overlayed_img = overlayed_img.astype(np.uint8)
                st.image(overlayed_img, caption=f"{file.name} → Grad-CAM Overlay", use_column_width=True)
            else:
                st.warning("No Conv2D layer found for Grad-CAM.")
        else:
            st.image(img, caption=f"{file.name} → Real", use_column_width=True)
