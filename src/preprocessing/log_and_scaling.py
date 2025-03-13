import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def apply_log_transformation(df, columns):
    """Apply log transformation to specified columns."""
    for col in columns:
        df[col] = df[col].apply(lambda x: np.log(x + 1) if x > 0 else 0)
    return df

def apply_scaling(df, numerical_features):
    """Apply Min-Max Scaling to numerical features except Age, and Standard Scaling to Age."""
    scaler = MinMaxScaler()
    df[numerical_features[1:]] = scaler.fit_transform(df[numerical_features[1:]])
    
    scaler_age = StandardScaler()
    df['Age'] = scaler_age.fit_transform(df[['Age']])
    
    return df

if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv('data/cleaned/cleaned_standardized_phone_usage.csv')
    
    # Numerical features for transformations
    numerical_features = [
        'Age', 'Screen Time (hrs/day)', 'Data Usage (GB/month)', 'Calls Duration (mins/day)',
        'Number of Apps Installed', 'Social Media Time (hrs/day)', 'E-commerce Spend (INR/month)',
        'Streaming Time (hrs/day)', 'Gaming Time (hrs/day)', 'Monthly Recharge Cost (INR)', 'Battery Consumption (mAh/day)'
    ]
    
    # Apply transformations
    df = apply_log_transformation(df, numerical_features[1:])
    df = apply_scaling(df, numerical_features)
    
    # Save processed data
    df.to_csv('data/processed/log_scaled_data.csv', index=False)
    print("Log transformation and scaling applied. Data saved as 'data/processed/log_scaled_data.csv'.")
