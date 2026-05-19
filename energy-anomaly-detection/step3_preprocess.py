import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def preprocess_data():
    if not os.path.exists('data/energy_data.csv'):
        print("❌ Error: Dataset not found. Run step1 first.")
        return

    df = pd.read_csv('data/energy_data.csv', parse_dates=['timestamp'])
    
    # Feature Engineering
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # Lag features (previous hour usage)
    df['usage_lag1'] = df['usage_kwh'].shift(1)
    df.dropna(inplace=True) # First row will have NaN lag
    
    # Define features and target
    features = ['usage_kwh', 'temperature', 'is_weekend', 'hour', 'day_of_week', 'month', 'usage_lag1']
    X = df[features]
    y = df['anomaly']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save processed data and scaler
    processed_data = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': features
    }
    
    joblib.dump(processed_data, 'data/processed_data.pkl')
    joblib.dump(scaler, 'data/scaler.pkl')
    
    print("✅ Preprocessing complete.")
    print(f"📏 Features used: {features}")
    print(f"📦 Data saved to 'data/processed_data.pkl'")

if __name__ == "__main__":
    preprocess_data()

