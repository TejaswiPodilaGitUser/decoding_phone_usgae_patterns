import streamlit as st
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eda import plots # Importing the plotting functions from plots.py
import seaborn as sns
import matplotlib.pyplot as plt

# Load the processed dataset
df = pd.read_csv('data/cleaned/cleaned_standardized_phone_usage.csv')

def main():
    # Streamlit UI setup
    st.title('Exploratory Data Analysis (EDA) for Mobile Usage Patterns')

    # Display a sample of the dataset
    st.subheader('Dataset Overview')
    if st.checkbox('Show Raw Data', key="show_raw_data_eda"):
        st.write(df.head())

    # Create two columns for histograms
    col1, col2 = st.columns(2)

    # Histogram 1: Age Distribution
    with col1:
        st.subheader('Distribution of Age')
        fig, ax = plt.subplots()
        sns.histplot(df['Age'], kde=True, color='skyblue', ax=ax)
        st.pyplot(fig)

    # Histogram 2: Screen Time Distribution
    with col2:
        st.subheader('Distribution of Screen Time')
        fig, ax = plt.subplots()
        sns.histplot(df['Screen Time (hrs/day)'], kde=True, color='orange', ax=ax)
        st.pyplot(fig)

    # Create two more columns for histograms
    col1, col2 = st.columns(2)

    # Histogram 3: Data Usage Distribution
    with col1:
        st.subheader('Distribution of Data Usage')
        fig, ax = plt.subplots()
        sns.histplot(df['Data Usage (GB/month)'], kde=True, color='green', ax=ax)
        st.pyplot(fig)

    # Histogram 4: Monthly Recharge Cost Distribution
    with col2:
        st.subheader('Distribution of Monthly Recharge Cost')
        fig, ax = plt.subplots()
        sns.histplot(df['Monthly Recharge Cost (INR)'], kde=True, color='purple', ax=ax)
        st.pyplot(fig)

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

if __name__ == "__main__":
    main()
