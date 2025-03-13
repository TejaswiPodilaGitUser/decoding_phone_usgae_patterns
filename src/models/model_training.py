import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Set experiment name
mlflow.set_experiment('Phone_Usage_Pattern_Classification')

# Load the cleaned dataset
df = pd.read_csv('data/processed/feature_engineered_data.csv')

# Split data into features and target
X = df.drop(columns=['Primary Use'])  # Features
y = df['Primary Use']  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "KNeighbors Classifier": KNeighborsClassifier(),
    "Support Vector Classifier (SVC)": SVC(),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier()
}

# Train and evaluate models
def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = []
    best_model = None
    best_model_name = ""
    best_f1_score = 0
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            precision = report['accuracy']
            recall = report['macro avg']['recall']
            f1_score = report['macro avg']['f1-score']
            
            mlflow.log_param("Model", model_name)
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("F1 Score", f1_score)
            
            input_example = np.array(X_train[:1])
            mlflow.sklearn.log_model(model, model_name, input_example=input_example)
            
            results.append({
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1_score
            })
            
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_model = model
                best_model_name = model_name

    print(f"Best Model: {best_model_name} with F1 Score: {best_f1_score:.4f}")
    return best_model, best_model_name, results

# Train models
best_model, best_model_name, model_results = train_and_evaluate(models, X_train_scaled, X_test_scaled, y_train, y_test)

# Save the best model
joblib.dump(best_model, 'models/best_classification_model.pkl')

# Save results to CSV
results_df = pd.DataFrame(model_results)
results_df.to_csv('results/model_comparison_results.csv', index=False)

print("All models and results logged to MLflow and saved.")
