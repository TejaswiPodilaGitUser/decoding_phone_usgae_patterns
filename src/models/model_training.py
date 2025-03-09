import pandas as pd
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
import joblib
import mlflow
import mlflow.sklearn
import os

# Ensure the results directory exists
os.makedirs('results', exist_ok=True)

# Set experiment name
mlflow.set_experiment('Phone_Usage_Pattern_Classification')  # Experiment Name

# Load the cleaned dataset
df = pd.read_csv('data/processed/feature_engineered_data.csv')

# Split the data into features and target
X = df.drop(columns=['Primary Use'])  # Features
y = df['Primary Use']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the models
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

# Function to train and evaluate models
def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = {}
    best_model = None
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    best_f1_score = 0
    
    # Loop through each model
    for model_name, model in models.items():
        # Use the model name as the run name in MLflow
        with mlflow.start_run(run_name=model_name):  # Start a new MLflow run with model name
            # Train the model
            model.fit(X_train, y_train)
            
            # Predict on the test set
            y_pred = model.predict(X_test)
            
            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            precision = report['accuracy']
            recall = report['macro avg']['recall']
            f1_score = report['macro avg']['f1-score']
            
            # Log model and metrics to MLflow
            mlflow.log_param("Model", model_name)
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("F1 Score", f1_score)
            
            # Log model with an input example (ensure it has the correct format)
            input_example = X_train[0].reshape(1, -1)  # Example input (first row reshaped to 2D array)
            mlflow.sklearn.log_model(model, model_name, input_example=input_example)
            
            # Store results in dictionary
            results[model_name] = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1_score
            }
            
            # Print the results for each model
            print(f"Model: {model_name}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1_score:.4f}")
            print("="*50)
            
            # Track the best model based on multiple metrics
            if accuracy > best_accuracy and precision > best_precision and recall > best_recall and f1_score > best_f1_score:
                best_accuracy = accuracy
                best_precision = precision
                best_recall = recall
                best_f1_score = f1_score
                best_model = model
                best_model_name = model_name
        
    print(f"Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f}, Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1 Score: {best_f1_score:.4f}")
    
    return best_model, results

# Train and evaluate the models
best_model, model_results = train_and_evaluate(models, X_train_scaled, X_test_scaled, y_train, y_test)

# Save the best model to a file using joblib
joblib.dump(best_model, 'models/best_classification_model.pkl')

# Convert the results into a DataFrame with each metric in its own column
results_list = []
for model_name, metrics in model_results.items():
    result = {
        "Model": model_name,
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1 Score": metrics["F1 Score"]
    }
    results_list.append(result)

# Convert results to a DataFrame
results_df = pd.DataFrame(results_list)

# Save the results to a CSV file
results_df.to_csv('results/model_comparison_results.csv', index=False)

# Print confirmation
print("Results saved to 'results/model_comparison_results.csv'")
print("Model training and evaluation completed successfully!")
print(f"Best model saved as 'models/best_classification_model.pkl'.")
print("All models and results logged to MLflow and saved.")
