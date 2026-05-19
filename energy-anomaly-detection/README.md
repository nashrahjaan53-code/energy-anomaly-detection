# ⚡ Energy Consumption Anomaly Detection

A machine learning project to detect abnormal patterns in electricity usage (potential theft, meter tampering, or faults) using hourly smart meter data.

## 📁 Project Structure
- `data/`: Contains the synthetic dataset (`energy_data.csv`) and processed files.
- `plots/`: Visualizations from EDA and model evaluation.
- `step1_create_dataset.py`: Generates a synthetic dataset with realistic daily/weekly patterns and injected anomalies.
- `step2_eda.py`: Performs Exploratory Data Analysis and saves key trend plots.
- `step3_preprocess.py`: Feature engineering (hour, lag features) and data scaling.
- `step4_kfold_models.py`: Uses **K-Fold Cross-Validation** to compare Random Forest, Logistic Regression, and Gradient Boosting.
- `step5_best_model_evaluate.py`: Trains the top-performing model and generates final evaluation metrics (ROC-AUC, Confusion Matrix).

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute the scripts in numeric order:
   ```bash
   python step1_create_dataset.py
   python step2_eda.py
   python step3_preprocess.py
   python step4_kfold_models.py
   python step5_best_model_evaluate.py
   ```

## 📊 Features Used
- `usage_kwh`: Power consumed (current hour)
- `temperature`: External temperature (affects AC/Heating usage)
- `hour`: Time of day (0-23)
- `is_weekend`: Boolean for weekends
- `usage_lag1`: Usage from the previous hour (crucial for time-series context)

## 🏆 Model Comparison
The project compares multiple classifiers using the F1-Score to handle the imbalanced nature of anomalies (only ~2% of the data).
