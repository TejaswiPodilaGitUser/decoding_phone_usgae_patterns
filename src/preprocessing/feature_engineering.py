import pandas as pd

def standardize_text(df, columns):
    """Convert text columns to lowercase and strip spaces for consistency."""
    for col in columns:
        df[col] = df[col].str.lower().str.strip()
    return df

def label_encode_features(df):
    """Apply label encoding to categorical features."""
    gender_mapping = {'Female': 0, 'Male': 1, 'Other': 2}
    df['Gender'] = df['Gender'].map(gender_mapping)
    
    os_mapping = {'android': 0, 'ios': 1}
    df['OS'] = df['OS'].map(os_mapping)
    
    primary_use_mapping = {'Education': 0, 'Gaming': 1, 'Entertainment': 2, 'Social Media': 3, 'Work': 4}
    df['Primary Use'] = df['Primary Use'].map(primary_use_mapping)
    
    phone_brand_mapping = {
        'vivo': 0, 'realme': 1, 'nokia': 2, 'samsung': 3, 'xiaomi': 4,
        'oppo': 5, 'apple': 6, 'google pixel': 7, 'motorola': 8, 'oneplus': 9
    }
    df['Phone Brand'] = df['Phone Brand'].map(phone_brand_mapping)
    
    location_mapping = {
        'Mumbai': 0, 'Delhi': 1, 'Ahmedabad': 2, 'Pune': 3, 'Jaipur': 4, 
        'Lucknow': 5, 'Kolkata': 6, 'Bangalore': 7, 'Chennai': 8, 'Hyderabad': 9
    }
    df['Location'] = df['Location'].map(location_mapping)
    
    return df

if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv('data/processed/log_scaled_data.csv')
    
    # Drop User ID if exists
    df.drop(columns=['User ID'], inplace=True, errors='ignore')
    
    # Standardize text features
    text_features = ['OS', 'Phone Brand']
    df = standardize_text(df, text_features)
    
    # Apply label encoding
    df = label_encode_features(df)
    
    # Save processed data
    df.to_csv('data/processed/feature_engineered_data.csv', index=False)
    print("Label encoding applied. Data saved as 'data/processed/feature_engineered_data.csv'.")
