import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from kneed import KneeLocator

# Ensure results folder exists
os.makedirs('results', exist_ok=True)

# Load dataset
df = pd.read_csv('data/processed/feature_engineered_data.csv')

# Select relevant features for clustering
X = df.drop(columns=['Primary Use'])  # Assuming 'Primary Use' is the target variable

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📌 Elbow method to determine the best K for KMeans
sse = []
k_range = range(2, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

elbow = KneeLocator(k_range, sse, curve='convex', direction='decreasing').elbow
print(f"Optimal number of clusters (Elbow method): {elbow}")

# 📌 Optimize DBSCAN parameters (manually tuned)
dbscan_eps = 0.3  # Adjust this based on dataset
dbscan_min_samples = 5

# 📌 Define clustering models with optimized parameters
clustering_models = {
    "KMeans": KMeans(n_clusters=elbow, random_state=42, n_init=10),
    "Hierarchical Clustering": AgglomerativeClustering(n_clusters=elbow),
    "DBSCAN": DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples),
    "Gaussian Mixture Models": GaussianMixture(n_components=elbow, random_state=42),
    "Spectral Clustering": SpectralClustering(n_clusters=elbow, random_state=42)
}

# 📌 Function to train and evaluate clustering models
def train_and_evaluate_clustering(models, X_scaled):
    results = {}
    best_model = None
    best_score = -1
    best_model_name = ""

    fig, axes = plt.subplots(1, len(models), figsize=(20, 5))

    for i, (model_name, model) in enumerate(models.items()):
        print(f"Training {model_name}...")

        model.fit(X_scaled)
        labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X_scaled)

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Ignore noise in DBSCAN
        cluster_sizes = [np.sum(labels == cluster) for cluster in range(num_clusters)]

        silhouette = silhouette_score(X_scaled, labels) if num_clusters > 1 else -1
        davies_bouldin = davies_bouldin_score(X_scaled, labels) if num_clusters > 1 else -1
        calinski_harabasz = calinski_harabasz_score(X_scaled, labels) if num_clusters > 1 else -1

        results[model_name] = {
            'Silhouette Score': silhouette,
            'Davies-Bouldin Score': davies_bouldin,
            'Calinski-Harabasz Score': calinski_harabasz,
            'Number of Clusters': num_clusters,
            'Cluster Sizes': cluster_sizes
        }

        # 📌 Track best model based on highest Silhouette Score
        if silhouette > best_score:
            best_score = silhouette
            best_model = model
            best_model_name = model_name

        # 📌 Plot clustering results
        axes[i].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', marker='.')
        axes[i].set_title(f'{model_name} Clusters')

    plt.show()
    return results, best_model, best_model_name

# 📌 Train models and get best one
clustering_results, best_clustering_model, best_model_name = train_and_evaluate_clustering(clustering_models, X_scaled)

# 📌 Save clustering results to CSV
results_df = pd.DataFrame([{**{'Model': k}, **v} for k, v in clustering_results.items()])
results_df.to_csv('results/clustering_comparison_results.csv', index=False)

# 📌 Save best clustering model as a pickle file
with open('results/best_clustering_model.pkl', 'wb') as f:
    pickle.dump(best_clustering_model, f)

print(f"Best clustering model ({best_model_name}) saved as 'results/best_clustering_model.pkl'")

# 📌 Save dataset with cluster labels
best_cluster_labels = best_clustering_model.labels_ if hasattr(best_clustering_model, 'labels_') else best_clustering_model.predict(X_scaled)
df['Cluster_Label'] = best_cluster_labels
df.to_csv('data/processed/clustered_data.csv', index=False)

print("Clustered data saved to 'data/processed/clustered_data.csv'")
