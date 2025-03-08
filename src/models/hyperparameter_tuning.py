import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Load the dataset
df = pd.read_csv("data/processed/feature_engineered_data.csv")

# Assume 'Primary Use' is the target column
X = df.drop('Primary Use', axis=1)
y = df['Primary Use']

# Handle missing values
X = X.fillna(X.mean())  # Impute missing values with column means (or use other strategies)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Resample to handle class imbalance (if needed)
X_train_resampled, y_train_resampled = resample(X_train, y_train, replace=True, n_samples=len(y_train), random_state=42)

# Set up parameter grids for RandomizedSearchCV
param_grids = {
    'Logistic Regression': {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['liblinear', 'saga'],
    },
    'Decision Tree': {
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10],
    },
    'KNeighbors Classifier': {
        'n_neighbors': [3, 5, 10],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10],
        'subsample': [0.8, 1.0],
    }
}

# Define models for hyperparameter tuning
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'KNeighbors Classifier': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(random_state=42)
}

# Function to perform RandomizedSearchCV for each model
def tune_model(model_name, model, param_grid, X_train, y_train):
    print(f"Starting hyperparameter tuning for {model_name}...")
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=10,  # Number of random combinations to try
        cv=3,       # Cross-validation folds
        verbose=1,
        n_jobs=-1,  # Use all CPU cores for parallelism
        random_state=42
    )

    # Fit the random search model
    random_search.fit(X_train, y_train)
    
    # Get the best model and hyperparameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    # Return the best model and params
    return best_model, best_params

# Function to evaluate the best model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Return the metrics
    return accuracy, precision, recall, f1

# Perform hyperparameter tuning for each model
best_models = {}
model_results = {}

for model_name, model in models.items():
    best_model, best_params = tune_model(model_name, model, param_grids[model_name], X_train_resampled, y_train_resampled)
    best_models[model_name] = best_model
    model_results[model_name] = best_params

    # Evaluate the best model
    accuracy, precision, recall, f1 = evaluate_model(best_model, X_test, y_test)
    model_results[model_name]['Accuracy'] = accuracy
    model_results[model_name]['Precision'] = precision
    model_results[model_name]['Recall'] = recall
    model_results[model_name]['F1 Score'] = f1

    print(f"Results for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    print(f"Best Parameters: {best_params}\n")

# Prepare the results for saving in CSV
results = []
for model_name, params in model_results.items():
    params['Model'] = model_name
    
    # Ensure the hyperparameters are in the correct order and included only if they exist for that model
    row = {
        'Model': model_name,
        'Accuracy': params['Accuracy'],
        'Precision': params['Precision'],
        'Recall': params['Recall'],
        'F1 Score': params['F1 Score'],
        'min_samples_split': params.get('min_samples_split', ''),
        'min_samples_leaf': params.get('min_samples_leaf', ''),
        'max_depth': params.get('max_depth', ''),
        'n_estimators': params.get('n_estimators', ''),
        'learning_rate': params.get('learning_rate', ''),
        'weights': params.get('weights', ''),
        'n_neighbors': params.get('n_neighbors', ''),
        'algorithm': params.get('algorithm', ''),
        'subsample': params.get('subsample', ''),
        'solver': params.get('solver', ''),
        'penalty': params.get('penalty', ''),
        'C': params.get('C', '')
    }
    results.append(row)

# Save the tuning results in a CSV
results_df = pd.DataFrame(results)
results_df.to_csv('results/hyperparameter_tuning_results.csv', index=False)

print("Hyperparameter tuning and evaluation complete!")
