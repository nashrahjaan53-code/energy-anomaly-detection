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
    
    # Define models to compare
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    print("\n" + "="*60)
    print("--- Model Comparison using 5-Fold Cross-Validation ---")
    print("="*60)
    
    # Setup K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    for name, model in models.items():
        # Using f1-score because the data is imbalanced (more normal than anomalies)
        cv_results = cross_val_score(model, X_train, y_train, cv=kf, scoring='f1')
        results[name] = cv_results
        
        print(f"\n{name}:")
        print(f"  F1-Scores per fold: {[f'{score:.4f}' for score in cv_results]}")
        print(f"  Mean F1-Score: {cv_results.mean():.4f}")
        print(f"  Std Deviation: {cv_results.std():.4f}")
    
    # Find best model
    best_model_name = max(results, key=lambda k: results[k].mean())
    best_score = results[best_model_name].mean()
    
    print("\n" + "="*60)
    print(f"🏆 WINNER: {best_model_name}")
    print(f"   Mean F1-Score: {best_score:.4f}")
    print("="*60 + "\n")
    joblib.dump(best_model_name, 'data/best_model_name.pkl')
    
    return best_model_name

if __name__ == "__main__":
    compare_models()