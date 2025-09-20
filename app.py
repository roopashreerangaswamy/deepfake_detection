# ----------------------------
# app.py - Deepfake Detection with Grad-CAM
# ----------------------------
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import gdown
import os
import matplotlib.pyplot as plt

# ----------------------------
# Constants
# ----------------------------
IMG_SIZE = (128, 128)
MODEL_PATH = "deepfake_detector.keras"  # Replace with your Google Drive file ID
GDRIVE_URL = f"https://drive.google.com/file/d/1LatU_PLtaXDuWUWlTOCHMFJ7eTOJGRQP/view?usp=drive_link"

# ----------------------------
# Download model if not exists
# ----------------------------
if not os.path.exists(MODEL_PATH):
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# ----------------------------
# Model definition
# ----------------------------
def build_model(input_shape=(128,128,3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Conv2D(128, (3,3), activation='relu')(x)
    last_conv_layer_output = x
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model, last_conv_layer_output

model, last_conv_layer_output = build_model()
model.load_weights(MODEL_PATH)
_ = model(tf.random.normal(shape=(1, IMG_SIZE[0], IMG_SIZE[1], 3)))  # Dummy call

# ----------------------------
# Grad-CAM
# ----------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_output):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer_output, model.output]
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
# Streamlit UI
# ----------------------------
st.title("🖼️ Deepfake Detection + Grad-CAM")
st.write("Upload one or more images to detect Real vs Fake content.")

uploaded_files = st.file_uploader("Choose images", type=['jpg','jpeg','png'], accept_multiple_files=True)

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
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_output)
            heatmap_resized = cv2.resize(heatmap, (img.width, img.height))
            heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap_resized), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            overlayed_img = np.array(img)*0.5 + heatmap_colored*0.5
            overlayed_img = overlayed_img.astype(np.uint8)
            st.image(overlayed_img, caption=f"{file.name} → Grad-CAM Overlay", use_column_width=True)
        else:
            st.image(img, caption=f"{file.name} → Real", use_column_width=True)
