import streamlit as st
import pandas as pd
import pickle
from plots_main import display_correlation_heatmap, display_boxplot, display_age_distribution, clustering_results
from sidebar import show_sidebar

@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned/cleaned_standardized_phone_usage_no_userid.csv")

@st.cache_resource
def load_model():
    with open('models/best_classification_model.pkl', 'rb') as file:
        return pickle.load(file)

def preprocess_input(input_data):
    """Convert categorical features to numeric encoding"""
    categorical_cols = ["Gender", "Location", "Phone Brand", "OS"]
    
    for col in categorical_cols:
        input_data[col] = pd.factorize(input_data[col])[0]  # Assign unique integer values
    
    return input_data

def main():
    st.title("📱 Phone Usage Prediction & Analysis")


    # Sidebar Input
    st.write("## 🔮 Prediction")
    user_input = show_sidebar()
    
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Encode categorical variables
    input_df = preprocess_input(input_df)

    # Check if Predict button was clicked
    if "predict" in st.session_state and st.session_state["predict"]:
        model = load_model()
        prediction = model.predict(input_df)
        st.success(f"📌 **Predicted Primary Use:** {prediction[0]}")
        st.session_state["predict"] = False  # Reset state

        
    # Load data
    data = load_data()
    st.write("### 📊 Data Preview")
    st.dataframe(data.head())

    # Display all plots
    st.write("## 📈 Data Insights")
    display_correlation_heatmap(data)
    display_boxplot(data)
    display_age_distribution(data)
    clustering_results(data)

if __name__ == "__main__":
    main()
