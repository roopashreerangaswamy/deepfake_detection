# ----------------------------
# app.py - Deepfake Detection + Grad-CAM
# ----------------------------
import streamlit as st
import numpy as np
from PIL import Image
import os

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Check if OpenCV is available
try:
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

# ----------------------------
# Constants
# ----------------------------
IMG_SIZE = (128, 128)
MODEL_PATH = "deepfake_detector.keras"

# ----------------------------
# Model definition (only if TensorFlow is available)
# ----------------------------
def build_model(input_shape=(128,128,3)):
    if not TF_AVAILABLE:
        return None, None
    
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

# Initialize model if possible
model, last_conv_layer_output = None, None
if TF_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        model, last_conv_layer_output = build_model()
        model.load_weights(MODEL_PATH)
        _ = model(tf.random.normal(shape=(1, IMG_SIZE[0], IMG_SIZE[1], 3)))  # Dummy call
    except Exception as e:
        st.error(f"Failed to load model: {e}")

# ----------------------------
# Grad-CAM (only if TensorFlow is available)
# ----------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_output):
    if not TF_AVAILABLE:
        return None
    
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

# Show status of dependencies
if not TF_AVAILABLE:
    st.warning("⚠️ TensorFlow not available. ML functionality is disabled.")
if not CV_AVAILABLE:
    st.warning("⚠️ OpenCV not available. Grad-CAM visualization is limited.")
if not os.path.exists(MODEL_PATH):
    st.warning("⚠️ Pre-trained model not found. Please upload the model file.")

if model is None:
    st.info("📋 Deepfake detection is currently unavailable. Upload functionality is still working.")

st.write("Upload one or more images to detect Real vs Fake content.")

uploaded_files = st.file_uploader("Choose images", type=['jpg','jpeg','png'], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        st.image(img, caption=f"Uploaded: {file.name}", use_column_width=True)
        
        if model is not None:
            img_array = np.expand_dims(np.array(img.resize(IMG_SIZE))/255.0, axis=0)
            
            # Prediction
            pred = model.predict(img_array)
            label = "Real" if pred[0][0] > 0.5 else "Fake"
            confidence = pred[0][0] if pred[0][0] > 0.5 else 1 - pred[0][0]
            st.write(f"**{file.name} → Prediction:** {label} (Confidence: {confidence:.2f})")

            # Grad-CAM overlay for Fake
            if label == "Fake" and CV_AVAILABLE:
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_output)
                if heatmap is not None:
                    heatmap_resized = cv2.resize(heatmap, (img.width, img.height))
                    heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap_resized), cv2.COLORMAP_JET)
                    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                    overlayed_img = np.array(img)*0.5 + heatmap_colored*0.5
                    overlayed_img = overlayed_img.astype(np.uint8)
                    st.image(overlayed_img, caption=f"{file.name} → Grad-CAM Overlay", use_column_width=True)
        else:
            st.info(f"📸 {file.name} uploaded successfully. ML analysis unavailable without model.")

# ----------------------------
# Instructions
# ----------------------------
st.markdown("---")
st.markdown("### 🔧 Setup Instructions")
st.markdown("""
To enable full functionality:
1. **Install TensorFlow**: The ML model requires TensorFlow
2. **Install OpenCV**: For Grad-CAM heatmap visualization
3. **Model File**: Upload or configure the deepfake_detector.keras model
""")

if not TF_AVAILABLE or not CV_AVAILABLE:
    st.code("pip install tensorflow opencv-python", language="bash")
