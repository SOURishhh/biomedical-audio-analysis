import os
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def generate_heart_sound(duration=6.0, sr=22050):
    """
    Synthesizes a realistic phonocardiogram (PCG) heartbeat sound:
    - S1 ('Lub'): Mitral and tricuspid valve closure (resonant ~120-150 Hz with acoustic body thump)
    - S2 ('Dub'): Aortic and pulmonary valve closure (slightly higher pitch ~180-220 Hz, shorter duration)
    - Realistic exponential acoustic decay and zero continuous static hiss.
    """
    total_samples = int(sr * duration)
    output = np.zeros(total_samples)
    t = np.arange(total_samples) / sr
    
    # Heart rate ~ 60-70 bpm (period ~ 0.9s)
    beat_period = 0.9
    num_beats = int(duration / beat_period)
    
    for beat in range(num_beats):
        t_s1 = beat * beat_period + 0.1
        t_s2 = t_s1 + 0.32  # Systole interval ~ 320ms
        
        # S1 ("Lub") component: ~130 Hz thump, duration ~ 120ms
        s1_len = int(0.12 * sr)
        t_s1_local = np.linspace(0, 0.12, s1_len)
        s1_carrier = np.sin(2 * np.pi * 130 * t_s1_local) + 0.4 * np.sin(2 * np.pi * 80 * t_s1_local)
        s1_decay = np.exp(-t_s1_local * 35) * np.sin(np.pi * t_s1_local / 0.12)
        s1_wave = s1_carrier * s1_decay * 0.8
        
        idx1 = int(t_s1 * sr)
        if idx1 + s1_len < total_samples:
            output[idx1 : idx1 + s1_len] += s1_wave
            
        # S2 ("Dub") component: ~190 Hz thump, duration ~ 80ms
        s2_len = int(0.09 * sr)
        t_s2_local = np.linspace(0, 0.09, s2_len)
        s2_carrier = np.sin(2 * np.pi * 190 * t_s2_local) + 0.3 * np.sin(2 * np.pi * 110 * t_s2_local)
        s2_decay = np.exp(-t_s2_local * 45) * np.sin(np.pi * t_s2_local / 0.09)
        s2_wave = s2_carrier * s2_decay * 0.6
        
        idx2 = int(t_s2 * sr)
        if idx2 + s2_len < total_samples:
            output[idx2 : idx2 + s2_len] += s2_wave
            
    # Add subtle physiological acoustic body reverberation
    output = butter_bandpass_filter(output, 40, 450, sr, order=2)
    # Normalize peak amplitude
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val * 0.85
    return output

def generate_lung_sound(duration=6.0, sr=22050):
    """
    Synthesizes a realistic respiratory sound (vesicular breath sound):
    - Smooth Inhalation (longer, wind-rush air turbulence centered ~300-900 Hz)
    - Brief pause
    - Exhalation (shorter, softer air turbulence ~200-600 Hz)
    - Zero flat white-noise hiss during breath pauses.
    """
    total_samples = int(sr * duration)
    output = np.zeros(total_samples)
    t = np.arange(total_samples) / sr
    
    # Respiratory cycle: ~3.0 seconds per full breath
    cycle_period = 3.0
    num_cycles = int(duration / cycle_period) + 1
    
    # Generate turbulence texture (pink-like spectrum)
    raw_noise = np.random.randn(total_samples)
    # Filter for realistic lung auscultation bandwidth (150 Hz to 1000 Hz)
    breath_sound = butter_bandpass_filter(raw_noise, 180, 950, sr, order=2)
    
    # Build breath envelope (inhale -> pause -> exhale -> pause)
    envelope = np.zeros(total_samples)
    for c in range(num_cycles):
        start_t = c * cycle_period
        # Inhale: 1.3 seconds
        in_start = int((start_t + 0.1) * sr)
        in_len = int(1.3 * sr)
        if in_start + in_len < total_samples:
            t_in = np.linspace(0, np.pi, in_len)
            envelope[in_start : in_start + in_len] = (np.sin(t_in) ** 1.8) * 0.9
            
        # Exhale: 1.0 second (after brief 0.2s pause)
        ex_start = int((start_t + 1.6) * sr)
        ex_len = int(1.0 * sr)
        if ex_start + ex_len < total_samples:
            t_ex = np.linspace(0, np.pi, ex_len)
            envelope[ex_start : ex_start + ex_len] = (np.sin(t_ex) ** 2) * 0.6
            
    output = breath_sound * envelope
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val * 0.85
    return output

def main():
    base_dir = "data"
    os.makedirs(os.path.join(base_dir, "Heart"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "Lung"), exist_ok=True)
    
    print("Generating acoustic Heart Sounds (audible PCG lub-dub)...")
    for i in range(1, 16):
        signal = generate_heart_sound()
        sf.write(os.path.join(base_dir, "Heart", f"heart_synth_{i}.wav"), signal, 22050)
        
    print("Generating acoustic Lung Sounds (audible respiratory cycle)...")
    for i in range(1, 16):
        signal = generate_lung_sound()
        sf.write(os.path.join(base_dir, "Lung", f"lung_synth_{i}.wav"), signal, 22050)
        
    print("Acoustic audio generation complete!")

if __name__ == "__main__":
    main()
