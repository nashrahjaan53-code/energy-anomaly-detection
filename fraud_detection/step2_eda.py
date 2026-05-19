import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset we created
df = pd.read_csv('fraud_data.csv')

# -----------------------------------------------
# 1. BASIC INFO — understand what we're working with
# -----------------------------------------------
print("=== Shape (rows, columns) ===")
print(df.shape)

print("\n=== First 5 rows ===")
print(df.head())

print("\n=== Data types ===")
print(df.dtypes)

print("\n=== Any missing values? ===")
print(df.isnull().sum())

print("\n=== Basic statistics ===")
print(df.describe())

# -----------------------------------------------
# 2. CLASS IMBALANCE — how many fraud vs legit?
# -----------------------------------------------
print("\n=== Fraud vs Legit count ===")
print(df['is_fraud'].value_counts())
print(f"Fraud %: {df['is_fraud'].mean() * 100:.2f}%")

plt.figure(figsize=(5, 4))
df['is_fraud'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'])
plt.title('Fraud vs Legit Transactions')
plt.xticks([0, 1], ['Legit', 'Fraud'], rotation=0)
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plot1_class_balance.png')
plt.show()
print("Saved: plot1_class_balance.png")

# -----------------------------------------------
# 3. AMOUNT DISTRIBUTION — do frauds have unusual amounts?
# -----------------------------------------------
plt.figure(figsize=(8, 4))
df[df['is_fraud'] == 0]['amount'].hist(bins=50, alpha=0.6, label='Legit', color='steelblue')
df[df['is_fraud'] == 1]['amount'].hist(bins=50, alpha=0.6, label='Fraud', color='tomato')
plt.title('Transaction Amount: Fraud vs Legit')
plt.xlabel('Amount')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('plot2_amount_distribution.png')
plt.show()
print("Saved: plot2_amount_distribution.png")

# -----------------------------------------------
# 4. HOUR OF DAY — when does fraud happen?
# -----------------------------------------------
plt.figure(figsize=(10, 4))
fraud_by_hour = df.groupby('hour')['is_fraud'].mean() * 100
fraud_by_hour.plot(kind='bar', color='tomato')
plt.title('Fraud Rate by Hour of Day')
plt.xlabel('Hour (0 = midnight)')
plt.ylabel('Fraud %')
plt.tight_layout()
plt.savefig('plot3_fraud_by_hour.png')
plt.show()
print("Saved: plot3_fraud_by_hour.png")

# -----------------------------------------------
# 5. CORRELATION HEATMAP — which features relate to fraud?
# -----------------------------------------------
plt.figure(figsize=(7, 5))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('plot4_correlation.png')
plt.show()
print("Saved: plot4_correlation.png")

# -----------------------------------------------
# 6. LOCATION MISMATCH — how much does it affect fraud?
# -----------------------------------------------
plt.figure(figsize=(5, 4))
df.groupby('location_mismatch')['is_fraud'].mean().plot(
    kind='bar', color=['steelblue', 'tomato']
)
plt.title('Fraud Rate: Location Mismatch')
plt.xticks([0, 1], ['Same Location', 'Different Location'], rotation=0)
plt.ylabel('Fraud Rate')
plt.tight_layout()
plt.savefig('plot5_location_mismatch.png')
plt.show()
print("Saved: plot5_location_mismatch.png")

print("\n=== EDA complete! Check all the saved plots ===")