import os
import pandas as pd
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.cluster import SpectralClustering, MeanShift, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv("data/processed/feature_engineered_data.csv")

# Assume 'Primary Use' is the target column
X = df.drop('Primary Use', axis=1)
y = df['Primary Use']

# Handle missing values
X = X.fillna(X.mean())

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define classification models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100, min_samples_leaf=1, max_features='sqrt'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1),
    'XGBoost': XGBClassifier(random_state=42, n_estimators=100, eval_metric='mlogloss'),
    'LightGBM': LGBMClassifier(random_state=42, n_estimators=100),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0, iterations=100)
}

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

# Function to train classification models
def train_classification_models():
    model_results = []
    best_model = None
    best_accuracy = 0
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        model_results.append([name, accuracy, precision, recall, f1])
        joblib.dump(model, f'models/{name.replace(" ", "_").lower()}.pkl')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    
    return best_model, model_results

# Train classification models
best_model, classification_results = train_classification_models()

# Save best classification model
joblib.dump(best_model, 'models/best_classification_clustering_model.pkl')

# Save classification results to CSV
os.makedirs('results', exist_ok=True)
results_df = pd.DataFrame(classification_results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results_df.to_csv('results/model_training_comparison_results.csv', index=False)

# Define clustering models
clustering_models = {
    'KMeans': KMeans(n_clusters=5, random_state=42, n_init=10),
    'Gaussian Mixture': GaussianMixture(n_components=5, random_state=42),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'Agglomerative Clustering': AgglomerativeClustering(n_clusters=5),
    'Spectral Clustering': SpectralClustering(n_clusters=5, random_state=42, assign_labels='discretize'),
    'Mean Shift': MeanShift(),
    'OPTICS': OPTICS(min_samples=5)
}


# Function to train clustering models
def train_clustering_models():
    clustering_results = []
    
    for name, model in clustering_models.items():
        print(f"Training {name}...")
        labels = model.fit_predict(X_scaled)
        
        silhouette = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else None
        davies_bouldin = davies_bouldin_score(X_scaled, labels) if len(set(labels)) > 1 else None
        calinski_harabasz = calinski_harabasz_score(X_scaled, labels) if len(set(labels)) > 1 else None
        
        clustering_results.append([name, silhouette, davies_bouldin, calinski_harabasz])
        joblib.dump(model, f'models/{name.replace(" ", "_").lower()}.pkl')
    
    return clustering_results

# Train clustering models
clustering_results = train_clustering_models()

# Save clustering results to CSV
clustering_results_df = pd.DataFrame(clustering_results, columns=['Model', 'Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score'])
clustering_results_df.to_csv('results/model_clustering_comparison_results.csv', index=False)

print("✅ Model training and saving complete! Results saved to 'results/model_training_comparison_results.csv' and 'results/model_clustering_comparison_results.csv'")
