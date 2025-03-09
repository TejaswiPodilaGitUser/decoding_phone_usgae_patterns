import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression


#  Data Preprocessing: Handling Missing Values

# Load your dataset
data = pd.read_csv('data/raw/phone_usage_india.csv')

# Function for handling missing values based on the discussed strategy
def handle_missing_values(df):
    # Create a dictionary to store missing values count before and after imputation
    missing_before = df.isnull().sum()

    # Handling 'Age' (Numerical - Predictive Imputation or Mean)
    if df['Age'].isnull().sum() > 0:
        # Predictive imputation (Linear Regression based on other features like Gender, Location)
        regressor = LinearRegression()
        features = ['Gender', 'Location']  # example features, adjust based on the actual data
        df_temp = df[features + ['Age']].dropna()  # Drop rows where 'Age' is not missing
        X = df_temp[features]
        y = df_temp['Age']
        regressor.fit(X, y)
        missing_data = df[df['Age'].isnull()]
        predicted_age = regressor.predict(missing_data[features])
        df.loc[df['Age'].isnull(), 'Age'] = predicted_age
    else:
        # Mean Imputation for Age if no regression is needed
        imputer = SimpleImputer(strategy='mean')
        df['Age'] = imputer.fit_transform(df[['Age']])

    # Handling 'Gender' (Categorical - Mode or Based on other features)
    if df['Gender'].isnull().sum() > 0:
        df['Gender'] = df.groupby('Location')['Gender'].transform(lambda x: x.fillna(x.mode()[0]))

    # Handling 'Location' (Categorical - Impute based on Region or Mode)
    if df['Location'].isnull().sum() > 0:
        df['Location'] = df['Location'].fillna(df['Location'].mode()[0])

    # Handling 'Phone Brand' (Categorical - Impute Based on OS)
    if df['Phone Brand'].isnull().sum() > 0:
        df['Phone Brand'] = df.groupby('OS')['Phone Brand'].transform(lambda x: x.fillna(x.mode()[0]))

    # Handling 'Screen Time' (Numerical - KNN Imputation)
    if df['Screen Time (hrs/day)'].isnull().sum() > 0:
        imputer = KNNImputer(n_neighbors=5)
        df['Screen Time (hrs/day)'] = imputer.fit_transform(df[['Screen Time (hrs/day)']])

    # Handling 'Data Usage' (Numerical - KNN Imputation)
    if df['Data Usage (GB/month)'].isnull().sum() > 0:
        imputer = KNNImputer(n_neighbors=5)
        df['Data Usage (GB/month)'] = imputer.fit_transform(df[['Data Usage (GB/month)']])

    # Handling 'Calls Duration' (Numerical - Mean Imputation)
    if df['Calls Duration (mins/day)'].isnull().sum() > 0:
        imputer = SimpleImputer(strategy='mean')
        df['Calls Duration (mins/day)'] = imputer.fit_transform(df[['Calls Duration (mins/day)']])

    # Handling 'Number of Apps Installed' (Numerical - KNN Imputation)
    if df['Number of Apps Installed'].isnull().sum() > 0:
        imputer = KNNImputer(n_neighbors=5)
        df['Number of Apps Installed'] = imputer.fit_transform(df[['Number of Apps Installed']])

    # Handling 'Social Media Time' (Numerical - KNN Imputation)
    if df['Social Media Time (hrs/day)'].isnull().sum() > 0:
        imputer = KNNImputer(n_neighbors=5)
        df['Social Media Time (hrs/day)'] = imputer.fit_transform(df[['Social Media Time (hrs/day)']])

    # Handling 'E-commerce Spend' (Numerical - Zero or Predictive Imputation)
    if df['E-commerce Spend (INR/month)'].isnull().sum() > 0:
        df['E-commerce Spend (INR/month)'] = df['E-commerce Spend (INR/month)'].fillna(0)

    # Handling 'Streaming Time' (Numerical - Mean Imputation)
    if df['Streaming Time (hrs/day)'].isnull().sum() > 0:
        imputer = SimpleImputer(strategy='mean')
        df['Streaming Time (hrs/day)'] = imputer.fit_transform(df[['Streaming Time (hrs/day)']])

    # Handling 'Gaming Time' (Numerical - Zero Imputation or Mean)
    if df['Gaming Time (hrs/day)'].isnull().sum() > 0:
        df['Gaming Time (hrs/day)'] = df['Gaming Time (hrs/day)'].fillna(0)  # Assuming no gaming activity

    # Handling 'Monthly Recharge Cost' (Numerical - KNN Imputation)
    if df['Monthly Recharge Cost (INR)'].isnull().sum() > 0:
        imputer = KNNImputer(n_neighbors=5)
        df['Monthly Recharge Cost (INR)'] = imputer.fit_transform(df[['Monthly Recharge Cost (INR)']])

    # Handling 'Primary Use' (Categorical - Predictive Imputation or Based on Related Features)
    if df['Primary Use'].isnull().sum() > 0:
        df['Primary Use'] = df.groupby('Phone Brand')['Primary Use'].transform(lambda x: x.fillna(x.mode()[0]))
        df['Primary Use'] = df['Primary Use'].fillna(df['Primary Use'].mode()[0])

    # Create a dictionary to store missing values count after imputation
    missing_after = df.isnull().sum()

    # Print how many missing values were handled
    for column in df.columns:
        handled = missing_before[column] - missing_after[column]
        if handled > 0:
            print(f"{handled} missing values handled in '{column}'")

    return df

# Apply the function to the data
cleaned_data = handle_missing_values(data)

# Save the cleaned data
cleaned_data.to_csv('data/cleaned/cleaned_phone_usage_india.csv', index=False)


# There are no missing values in the  data as per my analysis
print("Missing values handled successfully!")
