import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from kneed import KneeLocator
import os

# Load Dataset
df = pd.read_csv("data/processed/feature_engineered_data.csv")
X = df.drop(columns=['Primary Use']) if 'Primary Use' in df.columns else df.copy()

# Standardize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine Optimal K for KMeans
sse = []
k_range = range(2, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

elbow = KneeLocator(k_range, sse, curve="convex", direction="decreasing").elbow

# Define Clustering Models
clustering_models = {
    "KMeans": KMeans(n_clusters=elbow, random_state=42, n_init=10),
    "Hierarchical Clustering": AgglomerativeClustering(n_clusters=elbow),
    "DBSCAN": DBSCAN(eps=0.3, min_samples=5),
    "Gaussian Mixture": GaussianMixture(n_components=elbow, random_state=42),
    "Spectral Clustering": SpectralClustering(n_clusters=elbow, random_state=42)
}

# Train and Evaluate Models
results = []
for model_name, model in clustering_models.items():
    model.fit(X_scaled)
    labels = model.labels_ if hasattr(model, "labels_") else model.predict(X_scaled)
    
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    silhouette = silhouette_score(X_scaled, labels) if num_clusters > 1 else -1
    davies_bouldin = davies_bouldin_score(X_scaled, labels) if num_clusters > 1 else -1
    calinski_harabasz = calinski_harabasz_score(X_scaled, labels) if num_clusters > 1 else -1
    
    results.append({
        "Model": model_name,
        "Silhouette Score": round(silhouette, 4),
        "Davies-Bouldin Score": round(davies_bouldin, 4),
        "Calinski-Harabasz Score": round(calinski_harabasz, 4)
    })

# Save Results
results_df = pd.DataFrame(results)
os.makedirs("results", exist_ok=True)
results_df.to_csv("results/clustering_comparison_results.csv", index=False)

# Save All Models
os.makedirs("models", exist_ok=True)
for model_name, model in clustering_models.items():
    with open(f"models/clustering_model_{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)

# Save Best Model
best_model_name = max(results, key=lambda x: x["Silhouette Score"]) ["Model"]
with open(f"models/best_clustering_model_{best_model_name}.pkl", "wb") as f:
    pickle.dump(clustering_models[best_model_name], f)

print(f"Best Clustering Model: {best_model_name} (Saved in models/ directory)")
