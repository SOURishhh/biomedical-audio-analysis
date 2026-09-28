# Biomedical Audio Analysis 🩺

A machine learning and digital signal processing system for the classification and analysis of biomedical audio signals (**Heart Sounds vs. Lung Sounds**), based on the Summer Training Research Report (*Jawaharlal Nehru University & BPIT*).

---

## 🌟 Highlights & Novelty

1. **Authentic Biomedical Datasets**:
   - **Heart Sounds (Phonocardiogram - PCG)**: Sourced from the **PhysioNet / Computing in Cardiology Challenge** clinical benchmark.
   - **Lung Sounds (Respiratory Auscultation)**: Sourced from open clinical respiratory recordings containing full inhalation/exhalation breathing cycles.
2. **Comprehensive Acoustic Feature Engineering**:
   - Extracts 37 acoustic and spectral parameters described in **Table 3.7** of the research report:
     - 13 Mel-Frequency Cepstral Coefficients (**MFCCs**)
     - 12 Chroma Pitch-Class Energy bins
     - 7 Spectral Contrast bands
     - Root-Mean-Square Energy (**RMS**)
     - **Spectral Centroid** (frequency center of mass)
     - **Spectral Bandwidth** (frequency spread)
     - **Spectral Roll-Off** (upper 85% energy frequency)
     - **Zero-Crossing Rate** (ZCR)
3. **Multi-Model Benchmark & Ensemble Learning**:
   - Trains and benchmarks 5 standard supervised algorithms:
     - Logistic Regression (Linear baseline)
     - Decision Tree
     - K-Nearest Neighbors (KNN)
     - Support Vector Machine (SVM with RBF kernel)
     - Random Forest
   - **Novelty**: Includes a **Soft Voting Ensemble Classifier** combining Random Forest, SVM, and KNN to minimize variance and increase classification reliability.
4. **Interactive Auscultation Web App**:
   - Built with **Streamlit** to provide a real-time clinical inspection tool.
   - Listen to real heartbeats and breath sounds directly in the browser.
   - Live rendering of time-domain waveforms and Mel Spectrograms.
   - Real-time prediction confidence score distribution.
   - Summary table detailing the exact extracted feature values and their physiological significance.

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Real Data & Train Models
```bash
# Downloads PhysioNet PCG recordings and clinical respiratory files:
python download_real_data.py

# Trains all baseline models and the ensemble classifier:
python train.py
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```
Biomedical_Audio_Analysis/
├── app.py                      # Interactive Streamlit Web UI
├── train.py                    # End-to-end training and evaluation pipeline
├── download_real_data.py       # Clinical dataset downloader (PhysioNet PCG + Respiratory)
├── generate_synthetic_data.py  # Auscultation acoustic sound generator
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── data/
│   ├── Heart/                  # Real clinical heart sound recordings
│   └── Lung/                   # Real clinical lung breath sound recordings
├── models/
│   ├── best_model.pkl          # Trained Voting Classifier
│   ├── scaler.pkl              # Fitted feature standardizer
│   └── label_encoder.pkl       # Target label mapping
└── src/
    ├── preprocess.py           # Audio normalization, silence trimming, and padding
    ├── features.py             # Librosa feature extraction pipeline
    ├── data_loader.py          # Dataset matrix assembly (X, y)
    └── train_models.py         # Model training and validation logic
```

## 📜 Academic Attribution
Research internship carried out at the **School of Computer and Systems Sciences, Jawaharlal Nehru University (JNU), New Delhi**.
- **Student**: Sourish Surya
- **Supervisor**: Dr. Yogendra Meena
- **Institution**: Bhagwan Parshuram Institute of Technology (BPIT, GGSIPU)
