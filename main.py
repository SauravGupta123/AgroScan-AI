import streamlit as st
import cv2 as cv
import numpy as np
import keras

# Disease Labels
label_name = [
    'Apple Scab', 'Apple Black Rot', 'Apple Cedar Apple Rust', 'Apple Healthy', 'Cherry Powdery Mildew',
    'Cherry Healthy', 'Corn Cercospora Leaf Spot', 'Corn Common Rust', 'Corn Northern Leaf Blight', 'Corn Healthy',
    'Grape Black Rot', 'Grape Esca', 'Grape Leaf Blight', 'Grape Healthy', 'Peach Bacterial Spot',
    'Peach Healthy', 'Pepper Bell Bacterial Spot', 'Pepper Bell Healthy', 'Potato Early Blight',
    'Potato Late Blight', 'Potato Healthy', 'Strawberry Leaf Scorch', 'Strawberry Healthy',
    'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Late Blight', 'Tomato Leaf Mold', 
    'Tomato Septoria Leaf Spot', 'Tomato Spider Mites', 'Tomato Target Spot', 
    'Tomato Yellow Leaf Curl Virus', 'Tomato Mosaic Virus', 'Tomato Healthy'
]

# Load Model
@st.cache_resource
def load_model():
    return keras.models.load_model('Training/model/Leaf Deases(96,88).h5')

model = load_model()

# --- LANDING PAGE ---
st.markdown(
    """
    <style>
        .landing-container {
            text-align: center;
            margin-top: 50px;
        }
        .landing-title {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
        }
        .landing-subtitle {
            font-size: 20px;
            color: #666;
            margin-bottom: 30px;
        }
        .scroll-button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 18px;
            cursor: pointer;
            border-radius: 8px;
            transition: 0.3s;
        }
        .scroll-button:hover {
            background-color: #45a049;
        }
    </style>
    <div class="landing-container">
        <h1 class="landing-title">🌿 AI-Powered Leaf Disease Detection</h1>
        <p class="landing-subtitle">Identify diseases in plant leaves using advanced deep learning techniques.</p>
        <a href="#upload-section">
            <button class="scroll-button">📤 Upload Leaf Image</button>
        </a>
    </div>
    """, unsafe_allow_html=True
)

# --- UPLOAD SECTION ---
st.markdown('<h2 id="upload-section">📥 Upload a Leaf Image</h2>', unsafe_allow_html=True)
st.info("🔺 **Upload images of Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, or Tomato leaves.**")

uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read and process image
        image_bytes = uploaded_file.read()
        img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
        img_resized = cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150))
        normalized_image = np.expand_dims(img_resized, axis=0)

        # Display uploaded image
        st.image(img_resized, caption="📸 Uploaded Image", use_column_width=True)

        # Predict
        predictions = model.predict(normalized_image)
        confidence = predictions[0][np.argmax(predictions)] * 100
        predicted_label = label_name[np.argmax(predictions)]

        # Show result with confidence
        st.markdown(f"### 🟢 Prediction: **{predicted_label}**")
        st.progress(int(confidence))  # Show confidence as a progress bar
        st.markdown(f"**Confidence:** `{confidence:.2f}%`")

        if confidence < 80:
            st.warning("⚠️ The model is not very confident. Try another image.")
    
    except Exception as e:
        st.error(f"❌ Error: Could not process the image. {str(e)}")

# --- DESIGN & DEVELOPED BY SAURAV GUPTA ---
st.markdown(
    """
    <style>
        .developer-container {
            text-align: center;
            margin-top: 50px;
        }
        .developer-text {
            font-size: 18px;
            font-weight: bold;
            color: #444;
        }
        .social-icons {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 10px;
        }
        .social-icons a {
            text-decoration: none;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: bold;
        }
        .github { background-color: #333; color: white; }
        .linkedin { background-color: #0077B5; color: white; }
        .email { background-color: #D44638; color: white; }
        .github:hover { background-color: #555; }
        .linkedin:hover { background-color: #005f87; }
        .email:hover { background-color: #b23c2c; }
    </style>
    <div class="developer-container">
        <p class="developer-text">🚀 Designed & Developed by <strong>Saurav Gupta</strong></p>
        <div class="social-icons">
            <a class="github" href="https://github.com/SauravGupta123/" target="_blank">🔗 GitHub</a>
            <a class="linkedin" href="https://www.linkedin.com/in/srv-gupta/" target="_blank">🔗 LinkedIn</a>
            <a class="email" href="mailto:er.sauravgpt@gmail.com" target="_blank">📧 Email</a>
        </div>
    </div>
    """, unsafe_allow_html=True
)

# --- FOOTER (Full Width) ---
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #4CAF50;
            color: white;
            text-align: center;
            padding: 12px;
            font-size: 14px;
            font-weight: bold;
        }
    </style>
    <div class="footer">
        &copy; 2024 Leaf Disease Detection | All Rights Reserved 🌱
    </div>
    """, unsafe_allow_html=True
)
