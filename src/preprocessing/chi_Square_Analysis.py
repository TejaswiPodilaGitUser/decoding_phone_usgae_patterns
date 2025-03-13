import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

# Load dataset
df = pd.read_csv('data/cleaned/cleaned_phone_usage_india.csv')

# Drop User ID (not useful for correlation)
df.drop(columns=['User ID'], inplace=True)

# 1. Compute Correlation Matrix for Numerical Features
numerical_features = [
    'Age', 'Screen Time (hrs/day)', 'Data Usage (GB/month)', 'Calls Duration (mins/day)',
    'Number of Apps Installed', 'Social Media Time (hrs/day)', 'E-commerce Spend (INR/month)',
    'Streaming Time (hrs/day)', 'Gaming Time (hrs/day)', 'Monthly Recharge Cost (INR)'
]

# Compute correlation
correlation_matrix = df[numerical_features].corr()  # Only numeric columns


# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix (Numerical Features)")
plt.show()

# 2. Perform Chi-Square Test for Categorical Features
categorical_features = ['Gender', 'Location', 'Primary Use']

for col in categorical_features:
    contingency_table = pd.crosstab(df[col], df['Phone Brand'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-Square Test for {col} vs Phone Brand: p-value = {p:.5f}")

    contingency_table = pd.crosstab(df[col], df['OS'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-Square Test for {col} vs OS: p-value = {p:.5f}")

# 3. Mutual Information to Find Feature Importance
X = df[numerical_features]
y_brand = df['Phone Brand']
y_os = df['OS']

mi_brand = mutual_info_classif(X, y_brand, discrete_features=False)
mi_os = mutual_info_classif(X, y_os, discrete_features=False)

# Convert results to DataFrame
mi_results = pd.DataFrame({'Feature': numerical_features, 'MI_Brand': mi_brand, 'MI_OS': mi_os})
mi_results = mi_results.sort_values(by='MI_Brand', ascending=False)

print("\nMutual Information Scores:")
print(mi_results)
