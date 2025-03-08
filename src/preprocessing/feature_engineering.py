import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
for col in categorical_label_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save processed data
df.to_csv('data/processed/feature_engineered_data.csv', index=False)

print("Feature engineering completed with standardized formats and saved as 'data/processed/feature_engineered_data.csv'")
