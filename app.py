# app.py
import streamlit as st
from src.usecases.run_assessment import run_assessment
import tempfile

st.title("AthenaNet: Early Detection of Neuropsychiatric Disorders")

# User input
text_input = st.text_area("Enter your feelings/thoughts:")

audio_input = st.file_uploader("Upload an audio clip (wav)", type=["wav"])

if st.button("Assess"):
    if not text_input or not audio_input:
        st.error("Please provide both text and audio inputs.")
    else:
        # Save temporary audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_input.read())
            audio_path = tmp_file.name
        
        # Run prediction
        score = run_assessment(text_input, audio_path)
        st.success(f"Predicted Risk Score: {score:.3f}")
        if score > 0.5:
            st.warning("High risk detected! Consider seeking professional help.")
        else:
            st.info("Low risk detected. Keep monitoring your mental health.")
