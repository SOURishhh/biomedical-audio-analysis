import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate(X, y, save_dir='models'):
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # Add an ensemble model for novelty / better performance
    ensemble = VotingClassifier(
        estimators=[
            ('rf', models['Random Forest']),
            ('svm', models['SVM']),
            ('knn', models['KNN'])
        ],
        voting='soft'
    )
    models['Ensemble (RF+SVM+KNN)'] = ensemble
    
    results = {}
    best_model = None
    best_acc = 0
    
    os.makedirs(save_dir, exist_ok=True)
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        
        if acc >= best_acc:
            best_acc = acc
            best_model = model
            
    # Save the best model, scaler, and label encoder
    with open(os.path.join(save_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
        
    return results, best_model
