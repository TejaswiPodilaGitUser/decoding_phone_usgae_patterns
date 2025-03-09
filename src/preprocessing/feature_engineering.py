import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np


# Feature Engineering
# 1. Standardize 'OS' and 'Phone Brand' (convert to lowercase for consistency)
# 2. Label encode categorical features
# 3. Apply Log Transformation where needed
# 4. Apply Min-Max Scaling to numerical features except Age
# 5. Apply Standardization to Age


# Load preprocessed data (assuming missing values are handled and outliers are removed)
df = pd.read_csv('data/cleaned/cleaned_standardized_phone_usage.csv')

# Drop User ID as it's not useful for ML
df.drop(columns=['User ID'], inplace=True, errors='ignore')

# Standardize 'OS' and 'Phone Brand' (convert to lowercase for consistency)
df['OS'] = df['OS'].str.lower().str.strip()  # Standardize OS names (e.g., 'Android', 'android' -> 'android')
df['Phone Brand'] = df['Phone Brand'].str.lower().str.strip()  # Standardize phone models

# Define categorical features
categorical_label_features = ['Location', 'Phone Brand', 'Gender', 'OS', 'Primary Use']

# Label encode categorical features
label_encoders = {}

# Gender Encoding
gender_mapping = {'Female': 0, 'Male': 1, 'Other': 2}
df['Gender'] = df['Gender'].map(gender_mapping)

# OS Encoding
os_mapping = {'android': 0, 'ios': 1}  # Standardized OS names
df['OS'] = df['OS'].map(os_mapping)

# Primary Use Encoding
primary_use_mapping = {'Education': 0, 'Gaming': 1, 'Entertainment': 2, 'Social Media': 3, 'Work': 4}  # Standardized Primary Use
df['Primary Use'] = df['Primary Use'].map(primary_use_mapping)

# Phone Brand Encoding
phone_brand_mapping = {
    'vivo': 0, 'realme': 1, 'nokia': 2, 'samsung': 3, 'xiaomi': 4,
    'oppo': 5, 'apple': 6, 'google pixel': 7, 'motorola': 8, 'oneplus': 9}  # Standardized Phone Brands
df['Phone Brand'] = df['Phone Brand'].map(phone_brand_mapping)

# Location Encoding
location_mapping = {
    'Mumbai': 0, 'Delhi': 1, 'Ahmedabad': 2, 'Pune': 3, 'Jaipur': 4, 
    'Lucknow': 5, 'Kolkata': 6, 'Bangalore': 7, 'Chennai': 8, 'Hyderabad': 9}
df['Location'] = df['Location'].map(location_mapping)

# Numerical features for scaling and transformation
numerical_features = [
    'Age', 'Screen Time (hrs/day)', 'Data Usage (GB/month)', 'Calls Duration (mins/day)',
    'Number of Apps Installed', 'Social Media Time (hrs/day)', 'E-commerce Spend (INR/month)',
    'Streaming Time (hrs/day)', 'Gaming Time (hrs/day)', 'Monthly Recharge Cost (INR)', 'Battery Consumption (mAh/day)'
]

# Apply Log Transformation where needed
df['Screen Time (hrs/day)'] = df['Screen Time (hrs/day)'].apply(lambda x: np.log(x+1) if x > 0 else 0)
df['Data Usage (GB/month)'] = df['Data Usage (GB/month)'].apply(lambda x: np.log(x+1) if x > 0 else 0)
df['Calls Duration (mins/day)'] = df['Calls Duration (mins/day)'].apply(lambda x: np.log(x+1) if x > 0 else 0)
df['Number of Apps Installed'] = df['Number of Apps Installed'].apply(lambda x: np.log(x+1) if x > 0 else 0)
df['Social Media Time (hrs/day)'] = df['Social Media Time (hrs/day)'].apply(lambda x: np.log(x+1) if x > 0 else 0)
df['E-commerce Spend (INR/month)'] = df['E-commerce Spend (INR/month)'].apply(lambda x: np.log(x+1) if x > 0 else 0)
df['Streaming Time (hrs/day)'] = df['Streaming Time (hrs/day)'].apply(lambda x: np.log(x+1) if x > 0 else 0)
df['Gaming Time (hrs/day)'] = df['Gaming Time (hrs/day)'].apply(lambda x: np.log(x+1) if x > 0 else 0)
df['Monthly Recharge Cost (INR)'] = df['Monthly Recharge Cost (INR)'].apply(lambda x: np.log(x+1) if x > 0 else 0)
df['Battery Consumption (mAh/day)'] = df['Battery Consumption (mAh/day)'].apply(lambda x: np.log(x+1) if x > 0 else 0)

# Apply Min-Max Scaling to numerical features except Age
scaler = MinMaxScaler()
df[numerical_features[1:]] = scaler.fit_transform(df[numerical_features[1:]])

# Apply Standardization to Age
scaler_age = StandardScaler()
df['Age'] = scaler_age.fit_transform(df[['Age']])

# Save processed data
df.to_csv('data/processed/feature_engineered_data.csv', index=False)

print("Feature engineering completed with standardized formats, log transformation, and scaling applied. Data saved as 'data/processed/feature_engineered_data.csv'.")
