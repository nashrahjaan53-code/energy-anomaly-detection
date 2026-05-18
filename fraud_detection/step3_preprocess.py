import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle

df = pd.read_csv('fraud_data.csv')

X = df.drop(columns=['is_fraud'])   # all columns except the label
y = df['is_fraud']                  # only the label column

print("=== Features (X) shape:", X.shape)
print("=== Label (y) shape:", y.shape)
print("\nFeature columns:", list(X.columns))
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% for testing
    random_state=42,      # same split every run
    stratify=y            # keep same fraud % in both splits
)

print(f"\nTraining set size: {X_train.shape[0]} rows")
print(f"Test set size:     {X_test.shape[0]} rows")
print(f"\nFraud in training: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"Fraud in test:     {y_test.sum()} ({y_test.mean()*100:.1f}%)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn scale from train
X_test_scaled  = scaler.transform(X_test)         # apply same scale to test

print("\n=== Scaling done ===")
print("Mean of scaled training data (should be ~0):", X_train_scaled.mean().round(3))


print("\n=== Before SMOTE ===")
print(f"Legit: {(y_train == 0).sum()}, Fraud: {(y_train == 1).sum()}")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print("\n=== After SMOTE ===")
print(f"Legit: {(y_train_resampled == 0).sum()}, Fraud: {(y_train_resampled == 1).sum()}")
print("Now both classes are equal — model will learn both well!")

with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump({
        'X_train': X_train_resampled,
        'X_test':  X_test_scaled,
        'y_train': y_train_resampled,
        'y_test':  y_test,
        'scaler':  scaler,
        'feature_names': list(X.columns)
    }, f)

print("\n=== Preprocessing complete! Saved to preprocessed_data.pkl ===")