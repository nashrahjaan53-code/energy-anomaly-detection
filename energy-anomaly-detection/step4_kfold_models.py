import pandas as pd
import joblib
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import os

def compare_models():
    if not os.path.exists('data/processed_data.pkl'):
        print("❌ Error: Processed data not found. Run step3 first.")
        return

    data = joblib.load('data/processed_data.pkl')
    X_train = data['X_train']
    y_train = data['y_train']
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    print("\n--- Model Comparison using 5-Fold Cross-Validation ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    for name, model in models.items():
        cv_results = cross_val_score(model, X_train, y_train, cv=kf, scoring='f1')
        results[name] = cv_results
        print(f"{name}: Mean F1-Score = {cv_results.mean():.4f} (+/- {cv_results.std():.4f})")
    

    best_model_name = max(results, key=lambda k: results[k].mean())
    print(f"\n🏆 Best Model: {best_model_name}")
    joblib.dump(best_model_name, 'data/best_model_name.pkl')

if __name__ == "__main__":
    compare_models()
