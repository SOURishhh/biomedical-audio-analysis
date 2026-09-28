import os
from src.data_loader import build_feature_matrix
from src.train_models import train_and_evaluate
from generate_synthetic_data import main as generate_data

if __name__ == "__main__":
    if not os.path.exists("data/Heart") or len(os.listdir("data/Heart")) == 0:
        generate_data()
        
    print("Extracting features from data directory...")
    X, y, files = build_feature_matrix("data")
    
    if len(X) == 0:
        print("No data found. Exiting.")
    else:
        print(f"Extracted features for {len(X)} samples.")
        print("Training models...")
        results, best_model = train_and_evaluate(X, y)
        print("Training complete! Best model saved to 'models/' directory.")
