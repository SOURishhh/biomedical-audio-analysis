import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import pickle
import glob
import pandas as pd
from src.preprocess import load_and_preprocess
from src.features import extract_features

st.set_page_config(page_title="Biomedical Audio Classifier", page_icon="🩺", layout="wide")

st.title("🩺 Biomedical Audio Analysis & Auscultation Classifier")
st.markdown("""
This system analyzes biomedical audio recordings—specifically **heart sounds (phonocardiograms)** and **lung sounds (respiratory auscultation)**—using digital signal processing, acoustic feature engineering, and supervised machine learning.
""")

# Load models
@st.cache_resource
def load_models():
    model_path = os.path.join("models", "best_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    le_path = os.path.join("models", "label_encoder.pkl")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(le_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(le_path, 'rb') as f:
            le = pickle.load(f)
        return model, scaler, le
    return None, None, None

model, scaler, le = load_models()

if model is None:
    st.warning("Models not found! Please run `python train.py` first to generate models.")
    st.stop()

# Sidebar for choosing audio source
st.sidebar.header("📁 Audio Source")
source_option = st.sidebar.radio(
    "Choose how to load audio:",
    ("Benchmark Clinical Samples", "Upload Your Own Audio (.wav)")
)

audio_file_path = None
audio_label_hint = None

if source_option == "Benchmark Clinical Samples":
    heart_samples = sorted(glob.glob(os.path.join("data", "Heart", "*.wav")))
    lung_samples = sorted(glob.glob(os.path.join("data", "Lung", "*.wav")))
    
    category = st.sidebar.selectbox("Select Biomedical Category:", ("Heart Sound (PCG)", "Lung Sound (Respiratory)"))
    if category == "Heart Sound (PCG)" and heart_samples:
        selected_file = st.sidebar.selectbox("Choose a Heart Sample:", [os.path.basename(f) for f in heart_samples[:10]])
        audio_file_path = os.path.join("data", "Heart", selected_file)
        audio_label_hint = "Heart"
    elif category == "Lung Sound (Respiratory)" and lung_samples:
        selected_file = st.sidebar.selectbox("Choose a Lung Sample:", [os.path.basename(f) for f in lung_samples[:10]])
        audio_file_path = os.path.join("data", "Lung", selected_file)
        audio_label_hint = "Lung"
else:
    uploaded = st.sidebar.file_uploader("Upload Audio (.wav)", type=["wav"])
    if uploaded is not None:
        temp_path = "temp_user_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        audio_file_path = temp_path

if audio_file_path and os.path.exists(audio_file_path):
    st.subheader(f"Audio Sample: `{os.path.basename(audio_file_path)}`")
    
    # Audio Player
    with open(audio_file_path, "rb") as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format="audio/wav")
    
    # Process audio
    with st.spinner("Extracting biomedical acoustic features & running inference..."):
        signal, sr = load_and_preprocess(audio_file_path)
        features = extract_features(signal, sr)
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction_encoded = model.predict(features_scaled)
        prediction = le.inverse_transform(prediction_encoded)[0]
        
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features_scaled)[0]
            
    # Output Banner
    col_pred, col_conf = st.columns([1, 2])
    with col_pred:
        if prediction == "Heart":
            st.success("### ❤️ Result: Heart Sound")
        else:
            st.info("### 🫁 Result: Lung Sound")
            
    with col_conf:
        if probabilities is not None:
            st.write("**Model Classification Confidence:**")
            for i, class_name in enumerate(le.classes_):
                prob = probabilities[i] * 100
                st.progress(float(probabilities[i]), text=f"{class_name}: {prob:.1f}%")

    st.write("---")
    
    # Signal Visualizations
    st.subheader("📊 Acoustic Signal Representation")
    col1, col2 = st.columns(2)
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(7, 3.2))
        librosa.display.waveshow(signal, sr=sr, ax=ax1, color="#1f77b4")
        ax1.set_title("Preprocessed Time-Domain Waveform")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude (Normalized)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig1)
        
    with col2:
        fig2, ax2 = plt.subplots(figsize=(7, 3.2))
        S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, fmax=4000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=4000, ax=ax2, cmap="magma")
        fig2.colorbar(img, ax=ax2, format='%+2.0f dB')
        ax2.set_title("Time-Frequency Mel Spectrogram")
        st.pyplot(fig2)
        
    # Extracted Descriptors Summary Table (from Table 3.7 of Report)
    st.write("---")
    st.subheader("🔬 Extracted Feature Vector Descriptors (Table 3.7)")
    
    feature_summary = {
        "Descriptor Family": [
            "MFCC (Mean of first 3 coefficients)", 
            "RMS Energy", 
            "Spectral Centroid (Hz)", 
            "Spectral Bandwidth (Hz)", 
            "Spectral Roll-off (Hz)", 
            "Zero-Crossing Rate (ZCR)"
        ],
        "Computed Value": [
            f"[{features[0]:.2f}, {features[1]:.2f}, {features[2]:.2f}]",
            f"{features[32]:.4f}",
            f"{features[33]:.1f} Hz",
            f"{features[34]:.1f} Hz",
            f"{features[35]:.1f} Hz",
            f"{features[36]:.4f}"
        ],
        "Physiological Significance": [
            "Timbre & acoustic envelope of auscultation",
            "Overall mechanical sound power / amplitude",
            "Center of mass of frequency spectrum",
            "Spectral dispersion & frequency spread",
            "High-frequency cutoff (95% energy threshold)",
            "Frequency sign-change rate & high-frequency turbulence"
        ]
    }
    st.table(pd.DataFrame(feature_summary))
    
    if os.path.exists("temp_user_audio.wav"):
        os.remove("temp_user_audio.wav")
else:
    st.info("👈 Please select a sample recording from the sidebar or upload a `.wav` file to begin.")
