# ⚡ Energy Consumption Anomaly Detection

A machine learning project to detect abnormal patterns in electricity usage (potential theft, meter tampering, or faults) using hourly smart meter data.

## 📁 Project Structure

```
energy-anomaly-detection/
├── data/                      # Processed data and artifacts
│   ├── energy_data.csv
│   ├── processed_data.pkl
│   ├── scaler.pkl
│   └── best_model_name.pkl
├── models/                    # Trained ML models
│   └── energy_anomaly_model.pkl
├── plots/                     # Visualizations
│   ├── eda_class_balance.png
│   ├── eda_time_series.png
│   ├── final_confusion_matrix.png
│   └── final_roc_auc.png
├── step1_create_dataset.py    # Generates synthetic dataset
├── step2_eda.py               # Exploratory data analysis
├── step3_preprocess.py        # Feature engineering & scaling
├── step4_kfold_models.py      # K-Fold model comparison
├── step5_best_models.py       # Final model training & evaluation
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Execute the Pipeline
Run the scripts in order:
```bash
python step1_create_dataset.py  # Generate synthetic data
python step2_eda.py             # Analyze data patterns
python step3_preprocess.py      # Prepare features
python step4_kfold_models.py    # Compare models
python step5_best_models.py     # Train & evaluate
```

## 📊 Features Used

- **usage_kwh**: Power consumed (current hour)
- **temperature**: External temperature (affects AC/Heating usage)
- **is_weekend**: Binary flag for weekends
- **hour**: Time of day (0-23)
- **day_of_week**: Day of week (0-6)
- **month**: Month of year (1-12)
- **usage_lag1**: Usage from previous hour (crucial for anomaly detection)

## 🏆 Model Performance

### K-Fold Cross-Validation Results (5-Fold)
- **Logistic Regression**: 0.5630 F1-Score
- **Random Forest**: 0.8500 F1-Score
- **Gradient Boosting**: 0.8680 F1-Score ⭐ (Winner)

### Final Test Set Metrics
- **Precision**: 90% (avoids false alarms)
- **Recall**: 85% (catches most anomalies)
- **F1-Score**: 0.88 (balanced performance)

## 🔍 Key Insights

1. **Temporal Context Matters**: The `usage_lag1` feature (previous hour usage) is crucial for detecting sudden spikes/drops.

2. **Class Imbalance Handled**: Only ~2% of data contains anomalies, so F1-score and ROC-AUC are more reliable than accuracy.

3. **Gradient Boosting Wins**: Outperforms simpler models by learning complex patterns and corrections iteratively.

4. **Production Ready**: Model achieves 90% precision, meaning only 1 in 10 flagged cases is a false alarm.

## 📈 Example Output

```
Logistic Regression: Mean F1-Score = 0.5630 (+/- 0.1192)
Random Forest: Mean F1-Score = 0.8500 (+/- 0.0758)
Gradient Boosting: Mean F1-Score = 0.8680 (+/- 0.0401)

🏆 Best Model: Gradient Boosting

Classification Report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00      1719
           1       0.90      0.85      0.88        33
    accuracy                           1.00      1752
```

## 🛠️ Tech Stack

- **Data Processing**: Pandas, NumPy
- **ML Framework**: Scikit-Learn
- **Visualization**: Matplotlib, Seaborn
- **Model Serialization**: Joblib

## 📝 License

Open source - feel free to use and modify for your projects.

---

**Created as part of a Machine Learning exploration project.**
