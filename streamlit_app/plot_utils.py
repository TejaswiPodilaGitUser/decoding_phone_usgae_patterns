import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

def preprocess_data_for_corr(df):
    """Convert categorical columns to numeric for correlation calculations"""
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = pd.factorize(df[col])[0]  # Converts categorical to numeric
    return df

def display_correlation_heatmap(df):
    """ Display a heatmap of correlations """
    st.write("### 🔥 Correlation Heatmap")
    
    # Preprocess data to handle categorical columns before calculating correlation
    df_processed = preprocess_data_for_corr(df)
    
    # Calculate correlation
    corr = df_processed.corr()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    # Rotate x-axis labels by 30 degrees
    plt.xticks(rotation=30)
    st.pyplot(fig)

def display_scatter_plot(df):
    """ Display scatter plot for Data Usage vs Screen Time """
    st.write("### 📊 Scatter Plot: Data Usage vs Screen Time")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Data Usage (GB/month)', y='Screen Time (hrs/day)', data=df, ax=ax)
    st.pyplot(fig)

def display_line_plot(df):
    """ Display a line plot for Calls Duration over Index """
    st.write("### 📉 Line Plot: Calls Duration over Index")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(x=df.index, y='Calls Duration (mins/day)', data=df, ax=ax)
    st.pyplot(fig)

def display_count_plot(df):
    """ Display a count plot for Primary Use """
    st.write("### 📊 Count Plot: Primary Use")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Primary Use', data=df, ax=ax)
    st.pyplot(fig)

def display_age_distribution(df):
    """ Display age distribution histogram """
    st.write("### 📊 Age Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df['Age'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

def display_histogram(df):
    """ Display histogram for Data Usage """
    st.write("### 📊 Histogram: Data Usage Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df['Data Usage (GB/month)'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)
