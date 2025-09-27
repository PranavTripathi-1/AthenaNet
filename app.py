# app.py
import streamlit as st
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from src.infrastructure.text_encoder import TextEncoder
from src.infrastructure.audio_encoder import AudioEncoder
from src.infrastructure.fusion_model import FusionClassifier
from src.infrastructure.predictor import Predictor
from src.usecases.run_assessment import run_full_assessment
import soundfile as sf

# ---------------------------
# Load models
# ---------------------------
st.set_page_config(page_title="AthenaNet - Early Neuropsychiatric Detection", layout="wide")
st.title("AthenaNet 🧠 - Early Neuropsychiatric Risk Detection")

model_dir = Path("models")
text_encoder_path = model_dir / "text_encoder.pt"
audio_encoder_path = model_dir / "audio_encoder.pt"
fusion_model_path = model_dir / "fusion_classifier.pt"

st.sidebar.header("Model Status")
try:
    text_encoder = TextEncoder(str(text_encoder_path))
    audio_encoder = AudioEncoder(str(audio_encoder_path))
    fusion_model = FusionClassifier(str(fusion_model_path))
    predictor = Predictor(text_encoder, audio_encoder, fusion_model)
    st.sidebar.success("Models loaded successfully ✅")
except Exception as e:
    st.sidebar.error(f"Error loading models: {e}")

# ---------------------------
# User Input
# ---------------------------
st.header("Assessment Inputs")
text_input = st.text_area("Enter your thoughts / text for assessment:", height=150)
audio_input = st.file_uploader("Upload a short voice recording (WAV format):", type=["wav"])

# ---------------------------
# Run Prediction
# ---------------------------
if st.button("Run Assessment"):
    if not text_input:
        st.warning("Please enter text input for assessment!")
    elif not audio_input:
        st.warning("Please upload an audio file!")
    else:
        # Save temporary audio
        audio_path = "temp_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_input.getbuffer())
        
        # Run assessment pipeline
        try:
            result = run_assessment(predictor, text_input, audio_path)
            st.subheader("Assessment Result")
            st.metric("Risk Score", f"{result*100:.2f}%")
            if result > 0.7:
                st.warning("⚠️ High risk detected. Consider consulting a professional.")
            elif result > 0.4:
                st.info("🟡 Moderate risk. Monitor your mental health closely.")
            else:
                st.success("🟢 Low risk. Keep up healthy practices!")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
