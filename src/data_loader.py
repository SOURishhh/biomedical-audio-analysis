import os
import pandas as pd
import numpy as np
from src.preprocess import load_and_preprocess
from src.features import extract_features

def build_feature_matrix(data_dir):
    """
    Reads all audio files in the subdirectories of data_dir,
    extracts features, and creates the feature matrix X and labels y.
    Subdirectory names act as class labels.
    """
    features_list = []
    labels = []
    file_names = []
    
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                if file.endswith('.wav'):
                    file_path = os.path.join(class_dir, file)
                    try:
                        signal, sr = load_and_preprocess(file_path)
                        features = extract_features(signal, sr)
                        
                        features_list.append(features)
                        labels.append(class_name)
                        file_names.append(file)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        
    X = np.array(features_list)
    y = np.array(labels)
    
    return X, y, file_names
