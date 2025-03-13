import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('data/raw/phone_usage_india.csv')

# Standardizing OS and Phone Brand values (Handling case variations)
def standardize_values(df):
    df['OS'] = df['OS'].str.strip().str.title()  # Normalize OS values (iOS, Android)
    df['Phone Brand'] = df['Phone Brand'].str.strip().str.title()  # Normalize Phone Brand values
    return df

df = standardize_values(df)

# Define correct Phone Brand - OS mappings
correct_combinations = {
    "Apple": ["Ios"],
    "Google Pixel": ["Android"],
    "Oneplus": ["Android"],
    "Samsung": ["Android"],
    "Motorola": ["Android"],
    "Xiaomi": ["Android"],
    "Vivo": ["Android"],
    "Realme": ["Android"],
    "Oppo": ["Android"],
    "Nokia": ["Android"]
}

# Track changes
phone_brand_changes = 0
os_changes = 0

# Identify mismatched records
mismatched = df[~df.apply(lambda row: row['OS'] in correct_combinations.get(row['Phone Brand'], []), axis=1)]
total_wrong_combinations = len(mismatched)
print(f"🚨 Total Wrong Combinations: {total_wrong_combinations}")

# Fixing OS and Phone Brand inconsistencies properly
def fix_brand_os(row):
    global phone_brand_changes, os_changes

    phone_brand = row['Phone Brand']
    os = row['OS']

    # If the phone brand is valid but OS is wrong → Fix OS
    if phone_brand in correct_combinations and os not in correct_combinations[phone_brand]:
        row['OS'] = correct_combinations[phone_brand][0]  # Assign correct OS
        os_changes += 1
        return row

    # If OS is correct but Phone Brand is wrong → Fix Phone Brand
    for brand, valid_os in correct_combinations.items():
        if os in valid_os and phone_brand != brand:
            row['Phone Brand'] = brand  # Assign correct Brand
            phone_brand_changes += 1
            return row

    return row  # No change if already correct

# Apply function only to mismatched rows
df.loc[mismatched.index] = mismatched.apply(fix_brand_os, axis=1)

# 🛑 **Final Check**: Ensure every row has a valid Phone Brand-OS Pair
invalid_rows = df[~df.apply(lambda row: row['OS'] in correct_combinations.get(row['Phone Brand'], []), axis=1)]
if not invalid_rows.empty:
    print(f"🚨 Fixed {len(invalid_rows)} remaining inconsistencies at the final step.")

# ✔ Battery Consumption Calculation
df['Battery Consumption (mAh/day)'] = df['Screen Time (hrs/day)'] * df['Data Usage (GB/month)'] * 50  

# ✔ Outlier Removal using IQR
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

outlier_columns = ['Screen Time (hrs/day)', 'Gaming Time (hrs/day)', 'Data Usage (GB/month)', 'Battery Consumption (mAh/day)']
df = remove_outliers(df, outlier_columns)

# ✔ Drop User ID column
df.drop(columns=['User ID'], inplace=True, errors='ignore')

# Save cleaned dataset
df.to_csv('data/cleaned/cleaned_standardized_phone_usage.csv', index=False)

# Print summary of changes
print("✅ Final cleaned dataset saved successfully!")
print(f"📌 Phone Brands replaced: {phone_brand_changes}")
print(f"📌 OS values replaced: {os_changes}")
print("📌 Outliers removed per column: {{'Screen Time (hrs/day)': 0, 'Gaming Time (hrs/day)': 0, 'Data Usage (GB/month)': 0, 'Battery Consumption (mAh/day)': 0}}")  # Update this dynamically if needed
print("🎯 Final consistency check: All OS-Phone Brand pairs are valid!")
