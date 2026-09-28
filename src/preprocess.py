import librosa
import numpy as np

def load_and_preprocess(file_path, target_sr=22050, duration=5.0):
    """
    Loads an audio file, resamples it, trims silence, and pads/truncates 
    to a fixed duration to ensure consistent feature dimensions.
    """
    # Load audio
    signal, sr = librosa.load(file_path, sr=target_sr)
    
    # Trim leading/trailing silence
    signal, _ = librosa.effects.trim(signal, top_db=20)
    
    # Normalize amplitude
    signal = librosa.util.normalize(signal)
    
    # Fix length
    target_length = int(target_sr * duration)
    if len(signal) > target_length:
        signal = signal[:target_length]
    else:
        signal = np.pad(signal, (0, max(0, target_length - len(signal))), "constant")
        
    return signal, target_sr
