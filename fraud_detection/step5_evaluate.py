import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)


with open('trained_model.pkl', 'rb') as f:
    print("Loading trained model from trained_model.pkl...")
    model = pickle.load(f)

with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_test = data['X_test']
y_test = data['y_test']

# Get predictions
y_pred       = model.predict(X_test)           # hard prediction: 0 or 1
y_pred_proba = model.predict_proba(X_test)[:, 1]  # probability of fraud (0.0 to 1.0)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n=== Confusion Matrix ===")
print(f"True Negatives  (Legit correctly identified):  {tn}")
print(f"False Positives (Legit flagged as fraud):      {fp}")
print(f"False Negatives (Fraud missed by model):       {fn}  ← want this LOW")
print(f"True Positives  (Fraud correctly caught):      {tp}  ← want this HIGH")

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Legit', 'Fraud'],
    yticklabels=['Legit', 'Fraud']
)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('plot7_confusion_matrix.png')
plt.show()
print("Saved: plot7_confusion_matrix.png")

auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='tomato', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random guess')
plt.title('ROC-AUC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.legend()
plt.tight_layout()
plt.savefig('plot8_roc_auc.png')
plt.show()
print(f"\nROC-AUC Score: {auc:.4f}")
print("Saved: plot8_roc_auc.png")

ap = average_precision_score(y_test, y_pred_proba)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color='steelblue', lw=2, label=f'AP = {ap:.3f}')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall (fraud caught)')
plt.ylabel('Precision (correct fraud alerts)')
plt.legend()
plt.tight_layout()
plt.savefig('plot9_precision_recall.png')
plt.show()
print(f"Average Precision Score: {ap:.4f}")
print("Saved: plot9_precision_recall.png")

print("\n" + "="*45)
print("          FINAL MODEL SUMMARY")
print("="*45)
print(f"  Fraud cases in test set:   {y_test.sum()}")
print(f"  Fraud correctly caught:    {tp}  ({tp/y_test.sum()*100:.1f}% recall)")
print(f"  Fraud missed:              {fn}")
print(f"  False alarms raised:       {fp}")
print(f"  ROC-AUC Score:             {auc:.4f}")
print(f"  Average Precision:         {ap:.4f}")
print("="*45)

if auc > 0.90:
    print("Excellent model! AUC above 0.90.")
elif auc > 0.80:
    print("Good model! Try tuning for better recall.")
else:
    print("Needs improvement. Try SMOTE tuning or more features.")