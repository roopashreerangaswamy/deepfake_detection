import streamlit as st
from PIL import Image

st.set_page_config(page_title="Deepfake Detection Basic", page_icon="🔎")
st.title("🔎 Deepfake Detection - Basic App")

st.write("This is a minimal app to verify Streamlit deployment ✅")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.success("Image uploaded successfully!")
