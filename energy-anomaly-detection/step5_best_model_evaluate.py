import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import os

def evaluate_best_model():
    if not os.path.exists('data/processed_data.pkl') or not os.path.exists('data/best_model_name.pkl'):
        print("❌ Error: Files missing. Run steps 3 and 4 first.")
        return

    data = joblib.load('data/processed_data.pkl')
    best_name = joblib.load('data/best_model_name.pkl')
    
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    # Instantiate best model
    if best_name == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif best_name == 'Gradient Boosting':
        model = GradientBoostingClassifier(random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)
    
    print(f"\n--- Training Final {best_name} Model ---")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    # 1. Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {best_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('plots/final_confusion_matrix.png')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend()
    plt.savefig('plots/final_roc_auc.png')
    
    # Save final model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/energy_anomaly_model.pkl')
    
    print(f"✅ Final model saved to 'models/energy_anomaly_model.pkl'")
    print("📈 Evaluation plots saved in 'plots/' folder.")

if __name__ == "__main__":
    evaluate_best_model()
