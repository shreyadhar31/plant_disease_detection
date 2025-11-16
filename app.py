import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# BEAUTIFUL STREAMLIT UI SETUP
# ---------------------------
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #F5F9F3;
    }
    .title {
        color: #2E7D32;
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-top: -20px;
    }
    .subtitle {
        text-align: center;
        color: #4F5D4E;
        font-size: 18px;
        margin-bottom: 20px;
    }
    .result-box {
        padding: 20px;
        background: white;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .success {
        color: #1B5E20;
        font-size: 26px;
        font-weight: 700;
    }
    .label {
        color: #4E4E4E;
        font-size: 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# TITLE
# ---------------------------
st.markdown("<h1 class='title'>🌿 Plant Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a leaf image and let the AI predict the disease.</p>", unsafe_allow_html=True)

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("outputs/final_model.h5")  # change path if needed

model = load_model()

# Class names (edit based on your dataset)
class_names = [
    "Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy",
    "Potato Early Blight", "Potato Late Blight", "Potato Healthy",
    "Tomato Mosaic Virus", "Tomato Yellow Curl Virus", "Tomato Healthy"
]

# ---------------------------
# IMAGE UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("📸 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
st.image(img, caption="Uploaded Image", use_column_width=True)

st.markdown("### 🔄 Processing Image…")
with st.spinner("AI is analyzing the leaf. Please wait..."):

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100


    # ---------------------------
    # RESULT DISPLAY CARD
    # ---------------------------
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.markdown(f"<p class='label'>Prediction:</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='success'>{class_names[class_index]}</p>", unsafe_allow_html=True)

    st.markdown(f"### 🔢 Confidence: **{confidence:.2f}%**")
    st.markdown("</div>", unsafe_allow_html=True)
