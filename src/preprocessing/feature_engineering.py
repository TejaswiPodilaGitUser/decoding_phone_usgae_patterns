import pandas as pd

def standardize_text(df, columns):
    """Convert text columns to lowercase and strip spaces for consistency."""
    for col in columns:
        df[col] = df[col].fillna('unknown').str.lower().str.strip()
    return df

def label_encode_features(df):
    """Apply label encoding to categorical features with fallback for unknown values."""
    gender_mapping = {'female': 0, 'male': 1, 'other': 2}
    df['Gender'] = df['Gender'].str.lower().map(gender_mapping).fillna(-1)  # Handle unexpected values
    
    os_mapping = {'android': 0, 'ios': 1, 'unknown': -1}
    df['OS'] = df['OS'].map(os_mapping)
    
    primary_use_mapping = {'education': 0, 'gaming': 1, 'entertainment': 2, 'social media': 3, 'work': 4}
    df['Primary Use'] = df['Primary Use'].str.lower().map(primary_use_mapping).fillna(-1)
    
    phone_brand_mapping = {
        'vivo': 0, 'realme': 1, 'nokia': 2, 'samsung': 3, 'xiaomi': 4,
        'oppo': 5, 'apple': 6, 'google pixel': 7, 'motorola': 8, 'oneplus': 9,
        'unknown': -1
    }
    df['Phone Brand'] = df['Phone Brand'].map(phone_brand_mapping)
    
    location_mapping = {
        'mumbai': 0, 'delhi': 1, 'ahmedabad': 2, 'pune': 3, 'jaipur': 4, 
        'lucknow': 5, 'kolkata': 6, 'bangalore': 7, 'chennai': 8, 'hyderabad': 9,
        'unknown': -1
    }
    df['Location'] = df['Location'].str.lower().map(location_mapping).fillna(-1)
    
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
    print("✅ Label encoding applied successfully. Data saved as 'data/processed/feature_engineered_data.csv'.")
