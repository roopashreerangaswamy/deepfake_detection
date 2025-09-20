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
col1, col2 = st.columns([3, 1])

with col1:
    if not TF_AVAILABLE:
        st.warning("⚠️ TensorFlow not available. ML functionality is disabled.")
    if not CV_AVAILABLE:
        st.warning("⚠️ OpenCV not available. Grad-CAM visualization is limited.")
    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ Pre-trained model not found.")

with col2:
    if st.button("🔧 Setup Help"):
        st.info("See setup instructions below ⬇️")

# Model upload section
if TF_AVAILABLE and not os.path.exists(MODEL_PATH):
    st.markdown("### 📁 Upload Pre-trained Model")
    uploaded_model = st.file_uploader(
        "Upload your deepfake detection model (.keras format)", 
        type=['keras', 'h5'], 
        help="Upload a trained Keras model file for deepfake detection"
    )
    
    if uploaded_model is not None:
        # Save the uploaded model
        with open(MODEL_PATH, "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.success(f"Model '{uploaded_model.name}' uploaded successfully! Please refresh the page to load it.")
        st.experimental_rerun()

if model is None:
    st.info("📋 Deepfake detection is currently unavailable. Upload functionality is still working for image preview.")

st.markdown("### 📷 Image Analysis")
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

if not TF_AVAILABLE or not CV_AVAILABLE or not os.path.exists(MODEL_PATH):
    st.markdown("**Current Status:**")
    
    if TF_AVAILABLE:
        st.success("✅ TensorFlow is installed")
    else:
        st.error("❌ TensorFlow not installed")
        st.code("pip install tensorflow", language="bash")
    
    if CV_AVAILABLE:
        st.success("✅ OpenCV is installed") 
    else:
        st.error("❌ OpenCV not installed")
        st.code("pip install opencv-python", language="bash")
        
    if os.path.exists(MODEL_PATH):
        st.success("✅ Model file is available")
    else:
        st.error("❌ Model file missing")
        st.markdown("""
        **To get a pre-trained model:**
        1. Upload your own `.keras` model file using the uploader above
        2. Or train your own deepfake detection model
        3. The model should output a single value (0-1) where >0.5 = Real, <=0.5 = Fake
        """)
        
    st.markdown("---")
    st.markdown("**For developers:** This app supports any binary classification model trained for deepfake detection with 128x128 RGB input images.")
else:
    st.success("🎉 All dependencies are installed and model is loaded! The app is fully functional.")

# About section
with st.expander("ℹ️ About this Application"):
    st.markdown("""
    This is a **Deepfake Detection** application that uses:
    - **Deep Learning**: CNN-based binary classification 
    - **Grad-CAM**: Visual explanations showing which parts of the image influenced the decision
    - **Streamlit**: Web interface for easy image upload and analysis
    
    **How it works:**
    1. Upload an image (JPG, JPEG, or PNG)
    2. The model predicts if it's Real or Fake
    3. For images classified as Fake, Grad-CAM highlights suspicious regions
    
    **Note**: This is a demo application. For production use, consider additional preprocessing, ensemble methods, and validation techniques.
    """)
