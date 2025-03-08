import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the processed dataset
df = pd.read_csv('data/cleaned/cleaned_standardized_phone_usage.csv')

# 1. Trend in mobile app usage
def plot_app_usage_trends():
    plt.figure(figsize=(10, 6))
    df.groupby('Primary Use')['Screen Time (hrs/day)'].mean().sort_values().plot(kind='bar', color='lightseagreen')  # Changed color
    plt.title('Average Screen Time per Primary Use', fontsize=16)
    plt.xlabel('Primary Use', fontsize=12)
    plt.ylabel('Average Screen Time (hrs/day)', fontsize=12)
    plt.xticks(rotation=25)
    plt.tight_layout()
    return plt

# 2. Screen Time vs Data Usage
def plot_screen_time_vs_battery():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Screen Time (hrs/day)', y='Data Usage (GB/month)', data=df, color='darkorange')  # Changed color
    plt.title('Screen Time vs Data Usage', fontsize=16)
    plt.xlabel('Screen Time (hrs/day)', fontsize=12)
    plt.ylabel('Data Usage (GB/month)', fontsize=12)
    plt.tight_layout()
    return plt

# 3. Correlation Heatmap
def plot_battery_correlation():
    plt.figure(figsize=(10, 6))
    correlation = df[['Screen Time (hrs/day)', 'Data Usage (GB/month)', 'E-commerce Spend (INR/month)', 'Gaming Time (hrs/day)']].corr()
    sns.heatmap(correlation, annot=True, cmap='Blues', fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})  # Changed color map
    plt.title('Correlation Heatmap', fontsize=16)
    plt.xticks(rotation=25)
    plt.tight_layout()
    return plt

# 4. Primary Use Distribution
def plot_primary_use_patterns():
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Primary Use', data=df, palette='magma')  # Changed color palette
    plt.title('Primary Use Distribution', fontsize=16)
    plt.xlabel('Primary Use', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=25)
    plt.tight_layout()
    return plt

# 5. Data Usage by Primary Use
def plot_data_usage_by_primary_use():
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Primary Use', y='Data Usage (GB/month)', data=df, palette='coolwarm')  # Changed color palette
    plt.title('Data Usage by Primary Use', fontsize=16)
    plt.xlabel('Primary Use', fontsize=12)
    plt.ylabel('Data Usage (GB/month)', fontsize=12)
    plt.xticks(rotation=25)
    plt.tight_layout()
    return plt

# 6. Distribution of Battery Consumption
def plot_battery_consumption_histogram():
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Battery Consumption (mAh/day)'], kde=True, color='mediumslateblue')
    plt.title('Battery Consumption Distribution', fontsize=16)
    plt.xlabel('Battery Consumption (mAh/day)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    return plt

# 7. Monthly Recharge Cost Distribution
def plot_monthly_recharge_cost():
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Monthly Recharge Cost (INR)'], kde=True, color='forestgreen')
    plt.title('Monthly Recharge Cost Distribution', fontsize=16)
    plt.xlabel('Monthly Recharge Cost (INR)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    return plt

# 8. E-commerce Spend by Primary Use
def plot_ecommerce_spend_by_primary_use():
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Primary Use', y='E-commerce Spend (INR/month)', data=df, palette='inferno')  # Changed color palette
    plt.title('E-commerce Spend by Primary Use', fontsize=16)
    plt.xlabel('Primary Use', fontsize=12)
    plt.ylabel('E-commerce Spend (INR/month)', fontsize=12)
    plt.xticks(rotation=25)
    plt.tight_layout()
    return plt
