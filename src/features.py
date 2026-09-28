import librosa
import numpy as np

def extract_features(signal, sr):
    """
    Extracts acoustic and spectral features from an audio signal.
    Features: MFCC, Chroma, Spectral Contrast, RMS, Centroid, Bandwidth, Roll-off, ZCR
    """
    features = []
    
    # MFCC (13 coefficients)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfccs, axis=1))
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    
    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
    features.extend(np.mean(contrast, axis=1))
    
    # RMS Energy
    rms = librosa.feature.rms(y=signal)
    features.append(np.mean(rms))
    
    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    features.append(np.mean(centroid))
    
    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    features.append(np.mean(bandwidth))
    
    # Spectral Roll-off
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    features.append(np.mean(rolloff))
    
    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(signal)
    features.append(np.mean(zcr))
    
    return np.array(features)
