# Decoding Phone Usage Patterns in India

## Project Overview
This project aims to analyze mobile phone usage patterns in India through data preprocessing, feature engineering, machine learning classification, and clustering techniques. The results are visualized in a Streamlit app.



## Project Structure
```
📂 data
│   ├── cleaned
│   │   ├── cleaned_phone_usage_india.csv
│   │   ├── cleaned_standardized_phone_usage.csv
│   │   ├── cleaned_standardized_phone_usage_no_userid.csv
│   │   ├── gaming_screen_time_distribution.csv
│   │   └── gaming_time_distribution.csv
│   ├── processed
│   │   ├── cleaned_phone_usage.csv
│   │   ├── clustered_data.csv
│   │   ├── feature_engineered_data.csv
│   │   ├── labeled_data.csv
│   │   └── log_scaled_data.csv
│   └── raw
│       └── phone_usage_india.csv

📂 eda
│   ├── eda.py
│   ├── eda_processed.py
│   └── plots.py

📂 logs
│   └── mlflow

📂 models
│   ├── best_classification_model.pkl
│   ├── best_clustering_model_Spectral Clustering.pkl
│   ├── best_model.pkl
│   ├── classification_model.pkl
│   ├── clustering_model.pkl
│   ├── decision_tree.pkl
│   ├── gradient_boosting.pkl
│   ├── kneighbors_classifier.pkl
│   ├── lightgbm.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   ├── support_vector_classifier_(svc).pkl
│   └── xgboost.pkl

📂 notebooks
│   └── eda_phone_usage.ipynb

📂 requirements.txt

📂 results
│   ├── best_clustering_model.pkl
│   ├── clustering_comparison_results.csv
│   ├── clustering_optimized.csv
│   ├── clustering_results.csv
│   ├── hyperparameter_tuning_results.csv
│   ├── model_comparison.csv
│   └── model_comparison_results.csv

📂 scripts
│   ├── clustering.py
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   ├── hyperparameter_tuning.py
│   └── model_training.py

📂 src
│   ├── models
│   │   ├── clustering.py
│   │   ├── hyperparameter_tuning.py
│   │   └── model_training.py
│   ├── phone_usage_test.ipynb
│   └── preprocessing
│       ├── chi_Square_Analysis.py
│       ├── data_preprocessing.py
│       ├── data_standardization.py
│       ├── feature_engineering.py
│       └── log_and_scaling.py

📂 streamlit_app
│   ├── classification_page.py
│   ├── clustering_page.py
│   ├── eda_page.py
│   ├── main.py
│   ├── plot_utils.py
│   ├── plots_main.py
│   └── sidebar.py
```

## Features
- **Data Preprocessing & Feature Engineering**: Cleaning, scaling, and transforming the data for better model performance.
- **ML Model Training**: Training classification models for primary use prediction and clustering models for user segmentation.
- **Clustering Analysis**: Using KMeans, DBSCAN, Spectral Clustering, etc., to identify usage patterns.
- **Evaluation Metrics**: Using Silhouette Score, Davies-Bouldin Score, and Calinski-Harabasz Score for cluster evaluation.
- **Streamlit Dashboard**: Interactive visualizations of clustering results and data insights.

## Running the Project
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run data preprocessing:
   ```bash
   python scripts/data_preprocessing.py
   ```
3. Train clustering models:
   ```bash
   python scripts/clustering.py
   ```
4. Start the Streamlit app:
   ```bash
   streamlit run streamlit_app/main.py
   ```

## Results
- **Clustered Data Preview**: Displays the processed data with cluster labels.
- **Cluster Distribution**: Visual representation of the number of users in each cluster.
- **Cluster Visualization**: Scatter plot of clustered data for better pattern understanding.
- **Model Comparison**: Table displaying clustering model evaluation scores.

## Contributions
Feel free to contribute to the project by improving feature engineering, trying out new ML models, or enhancing the Streamlit app UI.

## License
This project is open-source and available under the MIT License.

