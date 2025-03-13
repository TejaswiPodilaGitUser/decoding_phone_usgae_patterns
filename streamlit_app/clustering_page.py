import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

def load_clustering_results():
    results_path = "results/clustering_comparison_results.csv"
    if os.path.exists(results_path):
        return pd.read_csv(results_path)
    else:
        st.error("Clustering results file not found. Please run clustering first.")
        st.stop()

def load_dataset():
    data_path = "data/processed/feature_engineered_data.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        st.error("Processed dataset file not found.")
        st.stop()

def load_model(model_name):
    model_path = f"models/clustering_model_{model_name}.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    else:
        st.error(f"Clustering model {model_name} file not found.")
        st.stop()

def plot_cluster_distribution(model_name, labels):
    fig, ax = plt.subplots(figsize=(5, 5))
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax, palette="viridis")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_name} - Cluster Distribution")
    st.pyplot(fig)

def plot_cluster_visualization(model_name, X, labels):
    fig, ax = plt.subplots(figsize=(5, 5))
    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap="viridis", marker="o", alpha=0.7, edgecolors='k')
    ax.set_title(f"{model_name} - Cluster Visualization")
    st.pyplot(fig)

def main():
    # Streamlit App Title
    st.title("📊 Phone Usage Clustering Results")
    
    # Load clustering results
    results_df = load_clustering_results()
    
    # Display Clustering Comparison Table
    st.write("### 🏆 Clustering Model Comparison")
    st.dataframe(results_df)
    
    # Load Dataset
    df = load_dataset()
    X = df.drop(columns=['Primary Use']) if 'Primary Use' in df.columns else df.copy()
    
    # Iterate over all models
    for model_name in results_df["Model"].unique():
        st.write(f"## {model_name} Clustering Results")
        model = load_model(model_name)
        labels = model.labels_ if hasattr(model, "labels_") else model.predict(X)
        df[f'Cluster_{model_name}'] = labels
        
        col1, col2 = st.columns(2)
        with col1:
            plot_cluster_distribution(model_name, labels)
        with col2:
            plot_cluster_visualization(model_name, X, labels)
    
    # Clustered Data Preview
    st.write("### 📋 Clustered Data Preview")
    preview_cols = ['Age', 'Gender', 'Phone Brand'] if {'Age', 'Gender', 'Phone Brand'}.issubset(df.columns) else df.columns[:3]
    st.dataframe(df[preview_cols + [f'Cluster_{model_name}']].head(10))
    
if __name__ == "__main__":
    main()