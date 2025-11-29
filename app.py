import streamlit as st
from PIL import Image
from utils.detection import load_model, predict, load_image_from_url
import os
import io

st.title("Egyptian Hieroglyphs Detection")

# Load model once
@st.cache_resource
def init_model():
    return load_model("models/detection.pt")

model = init_model()

# Input
st.subheader("Choose Input Method")

option = st.radio("Select input type:", ["Upload Image", "Image URL"])

img = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
else:
    url = st.text_input("Paste an image URL")
    if url:
        img = load_image_from_url(url)
        if img is None:
            st.error("Failed to load image from URL")

# Confidence threshold slider
confidence_threshold = st.slider(
    "confidence for detection", 
    min_value=0.0, max_value=1.0, value=0.3, step=0.05
)

# Detection 
if img:
    st.image(img, caption="Input Image", use_column_width=True)

    detections, output_img = predict(model, img, confidence_threshold)

    st.subheader("Detection Result")
    st.image(output_img, use_column_width=True)

    #Download button
    img_bytes = io.BytesIO()
    output_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    st.download_button(
        label="⬇️ Download Annotated Image",
        data=img_bytes,
        file_name="detection_result.png",
        mime="image/png"
    )

    st.write("Detections:", detections)
