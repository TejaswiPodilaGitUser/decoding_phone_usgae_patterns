import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv("data/processed/feature_engineered_data.csv")

# Assume 'Primary Use' is the target column
X = df.drop('Primary Use', axis=1)
y = df['Primary Use']

# Handle missing values
X = X.fillna(X.mean())  # Impute missing values with column means

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define parameter grids for RandomizedSearchCV
param_grids = {
    'Logistic Regression': {'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear', 'saga']},
    'Decision Tree': {'max_depth': [5, 10, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 5]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5, 10]},
    'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 10]},
    'KNeighbors Classifier': {'n_neighbors': [3, 5, 10], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
    'XGBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 10], 'subsample': [0.8, 1.0]}
}

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'KNeighbors Classifier': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(random_state=42)
}

# Function to perform RandomizedSearchCV
def tune_model(model_name, model, param_grid, X_train, y_train):
    print(f"Starting hyperparameter tuning for {model_name}...")
    random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=3, verbose=1, n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_, random_search.best_params_

# Function to evaluate models
def evaluate_model_with_cv(model, X, y):
    scores = {
        'Accuracy': cross_val_score(model, X, y, cv=5, scoring='accuracy').mean(),
        'Precision': cross_val_score(model, X, y, cv=5, scoring='precision_weighted').mean(),
        'Recall': cross_val_score(model, X, y, cv=5, scoring='recall_weighted').mean(),
        'F1 Score': cross_val_score(model, X, y, cv=5, scoring='f1_weighted').mean(),
        'ROC AUC': cross_val_score(model, X, y, cv=5, scoring='roc_auc_ovr_weighted').mean()
    }
    return scores

# Perform hyperparameter tuning
best_models = {}
model_results = {}

for model_name, model in models.items():
    best_model, best_params = tune_model(model_name, model, param_grids[model_name], X_train_resampled, y_train_resampled)
    best_models[model_name] = best_model
    best_params.update(evaluate_model_with_cv(best_model, X_test, y_test))
    model_results[model_name] = best_params
    print(f"Results for {model_name}: {best_params}\n")

# Save results to CSV
results_df = pd.DataFrame([{**{'Model': k}, **v} for k, v in model_results.items()])
os.makedirs('results', exist_ok=True)
results_df.to_csv('results/hyperparameter_tuning_results.csv', index=False)

# Save the best model
best_overall_model_name = max(model_results, key=lambda x: model_results[x]['Accuracy'])
best_overall_model = best_models[best_overall_model_name]
os.makedirs('models', exist_ok=True)
joblib.dump(best_overall_model, 'models/best_classification_model.pkl')
print(f"Best model '{best_overall_model_name}' saved as 'models/best_classification_model.pkl'")

print("Hyperparameter tuning and model saving complete!")
