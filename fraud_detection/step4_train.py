import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train        = data['X_train']
X_test         = data['X_test']
y_train        = data['y_train']
y_test         = data['y_test']
feature_names  = data['feature_names']

print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("\nTraining the model... (this may take a few seconds)")
model.fit(X_train, y_train)
print("Training done!")

train_acc = model.score(X_train, y_train) * 100
test_acc  = model.score(X_test,  y_test)  * 100

print(f"\nTraining Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy:     {test_acc:.2f}%")

if train_acc - test_acc > 10:
    print("Warning: possible overfitting! Try reducing max_depth.")
else:
    print("Looks good — no major overfitting detected.")

importances = model.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]  # sort high to low

print("\n=== Feature Importances (most useful → least) ===")
for i in sorted_idx:
    print(f"  {feature_names[i]:<25} {importances[i]:.4f}")

plt.figure(figsize=(7, 4))
plt.bar(
    [feature_names[i] for i in sorted_idx],
    importances[sorted_idx],
    color='steelblue'
)
plt.title('Feature Importances')
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('plot6_feature_importance.png')
plt.show()
print("Saved: plot6_feature_importance.png")

with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n=== Model saved to trained_model.pkl ===")
print("Run step5_evaluate.py next!")