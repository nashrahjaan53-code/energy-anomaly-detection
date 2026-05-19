# Fraud Detection System 🛡️

A modular end-to-end machine learning pipeline for detecting fraudulent transactions in imbalanced datasets.

## 📌 Project Overview
Fraud detection is a critical challenge in fintech. This project implements a robust classification system that identifies suspicious activities using a **Random Forest** model. It handles the classic "needle in a haystack" problem (highly imbalanced data) using **SMOTE** (Synthetic Minority Over-sampling Technique).

## 🚀 Technical Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, Imbalanced-Learn
- **Visualization:** Matplotlib, Seaborn
- **Model:** Random Forest Classifier

## 📂 Project Structure
- `step2_eda.py`: Exploratory Data Analysis & Visualization
- `step3_preprocess.py`: Data Cleaning, Scaling, and SMOTE balancing
- `step4_train.py`: Model training and Feature Importance analysis
- `step5_evaluate.py`: Performance metrics (ROC-AUC, Precision-Recall, Confusion Matrix)

## 📊 Key Features & Results
- **Handling Imbalance:** Applied SMOTE to ensure the model doesn't just "guess" the majority class.
- **Explainability:** Feature importance analysis revealed the top triggers for fraud (e.g., location mismatch, transaction amount).
- **Performance:** Evaluated using ROC-AUC and Precision-Recall curves to ensure high sensitivity (catching fraud) without excessive false alarms.

## 🛠️ How to Run
1. Clone the repository
2. Install dependencies: `pip install pandas scikit-learn imbalanced-learn matplotlib seaborn`
3. Run the scripts in order:
   ```bash
   python fraud_detection/step2_eda.py
   python fraud_detection/step3_preprocess.py
   python fraud_detection/step4_train.py
   python fraud_detection/step5_evaluate.py
   ```

---
*Created as part of a Machine Learning exploration project.*
