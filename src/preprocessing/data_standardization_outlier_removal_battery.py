import pandas as pd

# Step 1: Data Standardization - Phone Brand & OS
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
    return row['OS']  # Keep original if correct

# Apply OS correction
df['OS'] = df.apply(fix_os, axis=1)

# Remove rows where the Phone Brand is not in the valid list
df = df[df['Phone Brand'].isin(valid_os_mapping.keys())]

# Step 2: Adding Battery Consumption Based on Screen Time and Phone Brand
# Define battery consumption multiplier based on phone brand
battery_consumption_factors = {
    "Apple": 80,  # Example: Apple's battery consumption multiplier
    "Samsung": 70,  # Samsung's multiplier
    "Vivo": 60,  # Vivo's multiplier
    "Oppo": 60,  # Oppo's multiplier
    "Realme": 55,  # Realme's multiplier
    "OnePlus": 75,  # OnePlus multiplier
    "Xiaomi": 65,  # Xiaomi's multiplier
    "Google Pixel": 85,  # Google Pixel multiplier
    "Nokia": 50,  # Nokia's multiplier
    "Motorola": 55  # Motorola's multiplier
}

# Default multiplier for unlisted brands
default_multiplier = 50  # Adjust this value based on your needs

# Function to calculate battery consumption based on screen time and phone brand
def calculate_battery_consumption(row):
    # Get the multiplier for the phone brand, use default if brand is not found
    multiplier = battery_consumption_factors.get(row['Phone Brand'], default_multiplier)
    
    # Calculate battery consumption as screen time * multiplier
    return row['Screen Time (hrs/day)'] * multiplier

# Apply the function to add a new column for Battery Consumption
df['Battery Consumption (mAh/day)'] = df.apply(calculate_battery_consumption, axis=1)

# Step 3: Outlier Removal
# Function to remove outliers using IQR method
def remove_outliers(df):
    numerical_columns = ['Age', 'Screen Time (hrs/day)', 'Data Usage (GB/month)', 'Calls Duration (mins/day)', 
                         'Number of Apps Installed', 'Social Media Time (hrs/day)', 'E-commerce Spend (INR/month)', 
                         'Streaming Time (hrs/day)', 'Gaming Time (hrs/day)', 'Monthly Recharge Cost (INR)']
    
    for col in numerical_columns:
        if df[col].dtype in ['float64', 'int64']:  # Only handle numerical columns
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]  # Remove outliers

    return df

# Apply the function to remove outliers
df_cleaned = remove_outliers(df)

# Step 4: Drop User ID column
df_cleaned = df_cleaned.drop(columns=['User ID'])

# Save the final cleaned data with Battery Consumption, Outliers removed, and User ID dropped into a single CSV
df_cleaned.to_csv('data/cleaned/cleaned_standardized_phone_usage_no_userid.csv', index=False)

print("Data standardization, battery consumption addition, outliers removal, and User ID column drop completed successfully!")
