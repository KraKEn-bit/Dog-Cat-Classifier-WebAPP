import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# --- CONFIGURATION & THEME ---
st.set_page_config(page_title="PetScan AI", page_icon="🐾", layout="wide")

# Custom CSS for a sleek, modern look
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
    }
    /* Sleek cards for results */
    .prediction-card {
        background-color: #161b22;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        text-align: center;
    }
    .status-online {
        color: #238636;
        font-weight: bold;
        font-size: 0.9rem;
    }
    /* Glowing button effect */
    .stButton>button {
        width: 100%;
        background-image: linear-gradient(to right, #1f6feb, #111);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px #1f6feb;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: SYSTEM STATS ---
with st.sidebar:
    st.title("🐾 PetScan AI")
    st.markdown("---")
    st.header("📍 How to use")
    st.info("Upload any image of a cat or dog. Our Neural Network will analyze the features to determine the species.")
    
    st.header("🛠️ Specs")
    st.write("**Model:** Custom CNN")
    st.write("**Backend:** TensorFlow 2.16+")
    st.write("**Input Size:** 128x128px")
    
    st.header("👤 Author")
    st.caption("Developed by Rafsan Kabir")

# --- MAIN INTERFACE ---
st.title("Neural Vision: Dog vs Cat Classifier")
st.write("Experience the power of Deep Learning in real-time.")

# Load the model with error handling
@st.cache_resource
def load_classification_model():
    # Make sure this matches your new 'fixed' filename
    return load_model('dog_cat_fixed.keras')

try:
    model = load_classification_model()
    st.markdown('<p class="status-online">● SYSTEM STATUS: NEURAL ENGINE ONLINE</p>', unsafe_allow_html=True)
except Exception as e:
    st.error(f"Engine Failure: {e}")

# Creating a Two-Column Layout for Input vs Intelligence
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📁 Image Input")
    uploaded_file = st.file_uploader("Drop your image file here", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Analyzed Specimen", use_container_width=True)

with col_output:
    st.subheader("🧠 Intelligence Output")
    
    if uploaded_file is not None:
        # Preprocessing
        img_resized = img.resize((128, 128))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button("EXECUTE CLASSIFICATION"):
            with st.spinner('Neural Network processing...'):
                prediction = model.predict(img_array)[0][0]
                
                # Determine Label
                label = "DOG" if prediction > 0.5 else "CAT"
                confidence = prediction if prediction > 0.5 else 1 - prediction
                icon = "🐶" if label == "DOG" else "🐱"
                
                # Result Display
                st.markdown(f"""
                    <div class="prediction-card">
                        <h2 style="color: #1f6feb;">{icon} {label} DETECTED</h2>
                        <p style="font-size: 1.5rem;">Confidence Score: <b>{confidence*100:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Metrics and Charts
                st.write("---")
                m1, m2 = st.columns(2)
                m1.metric("Dog Likelihood", f"{prediction*100:.1f}%")
                m2.metric("Cat Likelihood", f"{(1-prediction)*100:.1f}%")
                
                st.bar_chart({"Probability": [1-prediction, prediction]}, x=["Cat", "Dog"])
    else:
        st.info("Waiting for image input to begin analysis...")