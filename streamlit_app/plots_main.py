import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Correlation Heatmap
def display_correlation_heatmap(df):
    col1, col2 = st.columns(2)

    with col1:
        st.write("## 🔥 Correlation Heatmap")
        df_numeric = df.select_dtypes(include=['number'])
        if not df_numeric.empty:
            corr = df_numeric.corr()
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
            st.pyplot(fig)
        else:
            st.write("No numerical columns available for correlation.")

    with col2:
        display_boxplot(df)  # Calling the boxplot function directly inside the second column


# Boxplot for multiple key variables
def display_boxplot(df):
    st.write("## 📊 Boxplot Analysis")
    
    features = ["Data Usage (GB/month)", "Screen Time (hrs/day)", "Calls Duration (mins/day)", "Social Media Time (hrs/day)"]
    
    feature = st.selectbox("Select Feature for Boxplot", features)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(x="Primary Use", y=feature, data=df, ax=ax, palette="Set2")
    plt.xticks(rotation=45)
    st.pyplot(fig)


# Age Distribution Plot
def display_age_distribution(df):
    col1, col2 = st.columns(2)

    with col1:
        st.write("## 🎂 Age Distribution")
        if 'Age' in df.columns and df['Age'].notnull().sum() > 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.histplot(df['Age'].dropna(), kde=True, bins=20, color='blue', ax=ax)
            plt.xlabel("Age")
            plt.ylabel("Frequency")
            plt.title("Age Distribution of Users")
            st.pyplot(fig)
        else:
            st.write("No valid Age data available.")

    with col2:
        clustering_results(df)  # Calling clustering function in the second column


# Clustering Analysis & Visualization
def clustering_results(df):
    st.write("## 🔍 Clustering Analysis")

    cluster_features = ['Screen Time (hrs/day)', 'Data Usage (GB/month)', 'Calls Duration (mins/day)', 'Social Media Time (hrs/day)']
    df_cluster = df[cluster_features].dropna()

    if df_cluster.shape[0] < 10:
        st.write("Not enough data points for clustering.")
        return

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters

    # Display cluster data
    st.write(df[['Age', 'Gender', 'Location', 'Primary Use', 'Cluster']].head())

    # Clustering Visualization
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=df['Screen Time (hrs/day)'], y=df['Data Usage (GB/month)'], hue=df['Cluster'], palette='Set1', ax=ax)
    plt.xlabel("Screen Time (hrs/day)")
    plt.ylabel("Data Usage (GB/month)")
    plt.title("Cluster Analysis: Screen Time vs. Data Usage")
    st.pyplot(fig)
