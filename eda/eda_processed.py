# eda_app.py

import streamlit as st
import pandas as pd
import plots  # Importing the plotting functions from plots.py

# Load the processed dataset
df = pd.read_csv('data/processed/feature_engineered_data.csv')

# Streamlit UI setup
st.set_page_config(page_title='Mobile Usage Patterns EDA', layout='wide')
st.title('Exploratory Data Analysis (EDA) for Mobile Usage Patterns')

# Display a sample of the dataset
st.subheader('Dataset Overview')
if st.checkbox('Show Raw Data'):
    st.write(df.head())

# Create two columns for the first row of plots
col1, col2 = st.columns(2)

# Plot 1: Trend in mobile app usage
with col1:
    st.subheader('Trends in Mobile App Usage')
    plot = plots.plot_app_usage_trends()  # Calling the function from plots.py
    st.pyplot(plot)

# Plot 2: Screen Time vs Data Usage
with col2:
    st.subheader('Screen Time vs Data Usage')
    plot = plots.plot_screen_time_vs_battery()  # Calling the function from plots.py
    st.pyplot(plot)

# Create two columns for the second row of plots
col1, col2 = st.columns(2)

# Plot 3: Correlation Heatmap
with col1:
    st.subheader('Correlation Heatmap')
    plot = plots.plot_battery_correlation()  # Calling the function from plots.py
    st.pyplot(plot)

# Plot 4: Primary Use Distribution
with col2:
    st.subheader('Primary Use Distribution')
    plot = plots.plot_primary_use_patterns()  # Calling the function from plots.py
    st.pyplot(plot)

# Create two columns for the third row of plots
col1, col2 = st.columns(2)

# Plot 5: Data Usage by Primary Use
with col1:
    st.subheader('Data Usage by Primary Use')
    plot = plots.plot_data_usage_by_primary_use()  # Calling the function from plots.py
    st.pyplot(plot)

# Plot 6: Battery Consumption Histogram
with col2:
    st.subheader('Battery Consumption Histogram')
    plot = plots.plot_battery_consumption_histogram()
    st.pyplot(plot)

# Create a row for the last two plots
col1, col2 = st.columns(2)

# Plot 7: Monthly Recharge Cost Distribution
with col1:
    st.subheader('Monthly Recharge Cost Distribution')
    plot = plots.plot_monthly_recharge_cost()  # Calling the function from plots.py
    st.pyplot(plot)

# Plot 8: E-commerce Spend by Primary Use
with col2:
    st.subheader('E-commerce Spend by Primary Use')
    plot = plots.plot_ecommerce_spend_by_primary_use()  # Calling the function from plots.py
    st.pyplot(plot)
