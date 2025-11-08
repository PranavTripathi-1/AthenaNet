🌿 AthenaNet – AI-Powered Mental Health Support System
📌 Overview

AthenaNet is an AI-powered mental health support system built using NLP and Deep Learning models to detect 
early signs of depression, anxiety, and stress from text and audio input. 
Unlike traditional chatbots, AthenaNet leverages transformer-based models and a lightweight Streamlit interface, making it both accessible and cost-effective.
This project aims to democratize mental health assistance, especially for students and underserved populations, by combining:
Real-time text/audio sentiment analysis
Adaptive AI-driven conversational support
A research-focused, open-source prototype

✨ Key Features

🧠 AI-based Detection – Uses NLP to identify depression, stress, and anxiety.
🎙️ Multi-modal Input – Accepts both text and audio (speech-to-text).
📊 Interactive Dashboard – Visualizes user sentiment in real time.
🔒 Privacy by Design – Local prototype with future scope for HIPAA/GDPR compliance.
🌍 Accessibility First – Runs on a lightweight Streamlit app deployable on any cloud.

🏗️ Project Architecture

Frontend: Streamlit Web App
Backend: Python (FastAPI/Flask for scalable deployment – future scope)
AI Models: Transformer-based NLP models (BERT, RoBERTa, DistilBERT)
Database (Optional): MongoDB/Postgres for storing anonymized user logs
Deployment: Streamlit Cloud (current), with scope for AWS/GCP scaling

🚀 Installation & Usage
1️⃣ Clone the Repository
git clone https://github.com/PranavTripathi-1/athenanet.git
cd athenanet

2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the App
streamlit run app.py

5️⃣ Access the Web App

Open your browser at:
👉 https://athenanet.streamlit.app/

📂 Project Structure
athenanet/
│── app.py                # Main Streamlit application
│── models/               # Pre-trained NLP models
│── utils/                # Helper functions (preprocessing, visualization, etc.)
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation
│── data/ (optional)      # Dataset storage for training/evaluation

🧪 Methodology

Data Preprocessing: Tokenization, cleaning, and feature extraction
Model Training: Fine-tuned transformer models for sentiment classification
Audio Input: Speech-to-text integration using Google Speech API / Whisper
Prediction & Visualization: Real-time classification with confidence scores
User Feedback Loop: Future enhancement for adaptive personalization

📊 Comparative Advantage

Unlike commercial apps such as Wysa or Woebot, 
AthenaNet:
Is open-source and research-driven
Focuses on early detection rather than just CBT-style chats
Provides multi-modal analysis (text + audio)
Prioritizes student accessibility and affordability

🔮 Future Scope

✅ Clinical validation with professionals
✅ Data encryption & compliance (HIPAA/GDPR)
✅ Integration of facial emotion recognition
✅ Adaptive coping strategy recommendations
✅ Cloud-native deployment with scalability

🏅 Acknowledgements

Developed as part of AI & ML Engineering coursework at IMS Engineering College, Ghaziabad
Inspired by the need to bridge gaps in student mental health support
Uses open-source libraries: Hugging Face Transformers, PyTorch, Streamlit

📜 License

This project is licensed under the MIT License – feel free to use and improve upon it with proper attribution.
