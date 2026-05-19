import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('plots', exist_ok=True)

def perform_eda():
    df = pd.read_csv('data/energy_data.csv', parse_dates=['timestamp'])
    
    plt.figure(figsize=(6, 4))
    sns.countplot(x='anomaly', data=df)
    plt.title('Normal vs Anomaly Counts')
    plt.savefig('plots/eda_class_balance.png')
    
    plt.figure(figsize=(12, 6))
    subset = df.head(24 * 7)
    plt.plot(subset['timestamp'], subset['usage_kwh'], alpha=0.7)
    anoms = subset[subset['anomaly'] == 1]
    plt.scatter(anoms['timestamp'], anoms['usage_kwh'], color='red')
    plt.title('Energy Usage - 1st Week')
    plt.savefig('plots/eda_time_series.png')
    
    print("✅ EDA complete. Plots saved in 'plots/'.")

if __name__ == "__main__":
    perform_eda()
