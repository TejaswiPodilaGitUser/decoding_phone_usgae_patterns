import pandas as pd


# Data Standardization, Outlier Removal, and Battery Consumption Addition
# Standardizing OS Values 
# Adding Battery Consumption Based on Screen Time and Phone Brand 
# Removing outliers 
# dropping User ID column


# Load dataset
df = pd.read_csv('data/cleaned/cleaned_phone_usage_india.csv')

# Define valid Phone Brand & OS mappings
valid_os_mapping = {
    "Apple": "iOS",
    "Samsung": "Android",
    "Vivo": "Android",
    "Oppo": "Android",
    "Realme": "Android",
    "OnePlus": "Android",
    "Xiaomi": "Android",
    "Google Pixel": "Android",
    "Nokia": "Android",
    "Motorola": "Android"
}

# Function to fix OS values
def fix_os(row):
    correct_os = valid_os_mapping.get(row['Phone Brand'])
    if correct_os and row['OS'] != correct_os:
        return correct_os  # Replace with correct OS
    return row['OS']  # Keep original if it's already correct

# Track how many OS values were changed
original_os_values = df['OS'].copy()
df['OS'] = df.apply(fix_os, axis=1)

# Count how many OS values were changed
changed_os_count = (df['OS'] != original_os_values).sum()

# Step 2: Adding Battery Consumption Based on Screen Time and Phone Brand
battery_consumption_factors = {
    "Apple": 80,
    "Samsung": 70,
    "Vivo": 60,
    "Oppo": 60,
    "Realme": 55,
    "OnePlus": 75,
    "Xiaomi": 65,
    "Google Pixel": 85,
    "Nokia": 50,
    "Motorola": 55
}

# Default multiplier for unlisted brands
default_multiplier = 50

# Function to calculate battery consumption based on screen time and phone brand
def calculate_battery_consumption(row):
    multiplier = battery_consumption_factors.get(row['Phone Brand'], default_multiplier)
    return row['Screen Time (hrs/day)'] * multiplier

# Apply the function to add a new column for Battery Consumption
df['Battery Consumption (mAh/day)'] = df.apply(calculate_battery_consumption, axis=1)

# Step 3: Outlier Removal
def remove_outliers(df):
    numerical_columns = ['Age', 'Screen Time (hrs/day)', 'Data Usage (GB/month)', 'Calls Duration (mins/day)', 
                         'Number of Apps Installed', 'Social Media Time (hrs/day)', 'E-commerce Spend (INR/month)', 
                         'Streaming Time (hrs/day)', 'Gaming Time (hrs/day)', 'Monthly Recharge Cost (INR)']
    
    removed_outliers = 0
    for col in numerical_columns:
        if df[col].dtype in ['float64', 'int64']:  # Only handle numerical columns
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            initial_count = df.shape[0]
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]  # Remove outliers
            removed_outliers += initial_count - df.shape[0]
            
            # Print debug info
            if initial_count - df.shape[0] > 0:
                print(f"Outliers removed from column: {col} | Removed: {initial_count - df.shape[0]} rows")

    return df, removed_outliers

# Apply the function to remove outliers and print how many rows were removed
df_cleaned, removed_outliers = remove_outliers(df)

# Step 4: Drop User ID column
df_cleaned = df_cleaned.drop(columns=['User ID'])

# Save the final cleaned data with Battery Consumption, Outliers removed, and User ID column dropped into a single CSV
df_cleaned.to_csv('data/cleaned/cleaned_standardized_phone_usage.csv', index=False)

# Print results
print(f"Data standardization, battery consumption addition, outliers removal, and User ID column drop completed successfully!")
print(f"Total OS values changed: {changed_os_count}")
print(f"Total outliers removed: {removed_outliers}")
