import streamlit as st

st.title("Deepfake Detector")
st.write("App is running!")

# Lazy-load the model only on user interaction
# e.g., when a file is uploaded
uploaded_file = st.file_uploader("Upload an image")
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image")
    st.write("You can now run prediction here...")

