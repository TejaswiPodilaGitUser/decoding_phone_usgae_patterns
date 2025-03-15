import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the processed dataset
df = pd.read_csv('data/cleaned/cleaned_standardized_phone_usage.csv')

# Define constants for column names
PRIMARY_USE = 'Primary Use'
SCREEN_TIME = 'Screen Time (hrs/day)'
DATA_USAGE = 'Data Usage (GB/month)'
E_COMMERCE_SPEND = 'E-commerce Spend (INR/month)'
BATTERY_CONSUMPTION = 'Battery Consumption (mAh/day)'
MONTHLY_RECHARGE = 'Monthly Recharge Cost (INR)'

# 1. Trend in mobile app usage
def plot_app_usage_trends():
    plt.figure(figsize=(10, 6))
    df.groupby(PRIMARY_USE)[SCREEN_TIME].mean().sort_values().plot(kind='bar', color='lightseagreen')
    plt.title('Average Screen Time per Primary Use', fontsize=16)
    plt.xlabel(PRIMARY_USE, fontsize=12)
    plt.ylabel(f'Average {SCREEN_TIME}', fontsize=12)
    plt.xticks(rotation=25)
    plt.tight_layout()
    return plt

# 2. Screen Time vs Data Usage
def plot_screen_time_vs_battery():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=SCREEN_TIME, y=DATA_USAGE, data=df, color='darkorange')
    plt.title(f'{SCREEN_TIME} vs {DATA_USAGE}', fontsize=16)
    plt.xlabel(SCREEN_TIME, fontsize=12)
    plt.ylabel(DATA_USAGE, fontsize=12)
    plt.tight_layout()
    return plt

# 3. Correlation Heatmap
def plot_battery_correlation():
    plt.figure(figsize=(10, 6))
    correlation = df[[SCREEN_TIME, DATA_USAGE, E_COMMERCE_SPEND, 'Gaming Time (hrs/day)']].corr()
    sns.heatmap(correlation, annot=True, cmap='Blues', fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Heatmap', fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

# 4. Primary Use Distribution
def plot_primary_use_patterns():
    plt.figure(figsize=(10, 6))
    sns.countplot(x=PRIMARY_USE, data=df, palette='magma')
    plt.title('Primary Use Distribution', fontsize=16)
    plt.xlabel(PRIMARY_USE, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=25)
    plt.tight_layout()
    return plt

# 5. Data Usage by Primary Use
def plot_data_usage_by_primary_use():
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=PRIMARY_USE, y=DATA_USAGE, data=df, palette='coolwarm')
    plt.title(f'{DATA_USAGE} by {PRIMARY_USE}', fontsize=16)
    plt.xlabel(PRIMARY_USE, fontsize=12)
    plt.ylabel(DATA_USAGE, fontsize=12)
    plt.xticks(rotation=25)
    plt.tight_layout()
    return plt

# 6. Distribution of Battery Consumption
def plot_battery_consumption_histogram():
    plt.figure(figsize=(10, 6))
    sns.histplot(df[BATTERY_CONSUMPTION], kde=True, color='mediumslateblue')
    plt.title('Battery Consumption Distribution', fontsize=16)
    plt.xlabel(BATTERY_CONSUMPTION, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    return plt

# 7. Monthly Recharge Cost Distribution
def plot_monthly_recharge_cost():
    plt.figure(figsize=(10, 6))
    sns.histplot(df[MONTHLY_RECHARGE], kde=True, color='forestgreen')
    plt.title('Monthly Recharge Cost Distribution', fontsize=16)
    plt.xlabel(MONTHLY_RECHARGE, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    return plt

# 8. E-commerce Spend by Primary Use
def plot_ecommerce_spend_by_primary_use():
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=PRIMARY_USE, y=E_COMMERCE_SPEND, data=df, palette='inferno')
    plt.title(f'{E_COMMERCE_SPEND} by {PRIMARY_USE}', fontsize=16)
    plt.xlabel(PRIMARY_USE, fontsize=12)
    plt.ylabel(E_COMMERCE_SPEND, fontsize=12)
    plt.xticks(rotation=25)
    plt.tight_layout()
    return plt
