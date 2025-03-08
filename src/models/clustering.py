import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data/processed/feature_engineered_data.csv')

# Select relevant features for clustering (you may adjust this based on your dataset)
X = df.drop(columns=['Primary Use'])  # Assuming 'Primary Use' is the target variable

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to train and evaluate clustering models
def train_and_evaluate_clustering(clustering_models, X_scaled):
    results = {}
    
    for model_name, model in clustering_models.items():
        print(f"Training {model_name}...")
        
        # Fit the model
        model.fit(X_scaled)
        
        # Get the labels assigned by the model
        labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X_scaled)
        
        # Calculate the silhouette score if there are more than one cluster
        if len(set(labels)) > 1:
            silhouette_avg = silhouette_score(X_scaled, labels)
        else:
            silhouette_avg = -1  # DBSCAN might return only one cluster or noise
        
        # Store results
        results[model_name] = {
            'Labels': labels,
            'Silhouette Score': silhouette_avg
        }
        
        # Print the results
        print(f"Silhouette Score for {model_name}: {silhouette_avg:.4f}")
        
        # Visualize the clusters if silhouette score is positive
        if silhouette_avg > 0:
            plt.figure(figsize=(8, 6))
            plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', marker='.')
            plt.title(f'{model_name} - Clusters Visualization')
            plt.show()
    
    return results

# Define the clustering models
clustering_models = {
    "KMeans": KMeans(n_clusters=3, random_state=42),  # Try different n_clusters if needed
    "Hierarchical Clustering": AgglomerativeClustering(n_clusters=3),
    "DBSCAN": DBSCAN(eps=0.3, min_samples=10),
    "Gaussian Mixture Models": GaussianMixture(n_components=3, random_state=42),  # Try different components if needed
    "Spectral Clustering": SpectralClustering(n_clusters=3, random_state=42)
}

# Train and evaluate clustering models
clustering_results = train_and_evaluate_clustering(clustering_models, X_scaled)

# Save clustering results to a CSV
results_list = []
for model_name, metrics in clustering_results.items():
    result = {
        "Model": model_name,
        "Silhouette Score": metrics["Silhouette Score"]
    }
    results_list.append(result)

# Convert results to a DataFrame
results_df = pd.DataFrame(results_list)

# Save the results to a CSV file
results_df.to_csv('results/clustering_comparison_results.csv', index=False)

# Print confirmation
print("Clustering results saved to 'results/clustering_comparison_results.csv'")
print("Clustering evaluation completed successfully!")
