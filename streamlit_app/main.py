# main.py
import streamlit as st
import pandas as pd
import joblib
from plots_main import display_all_plots
from sidebar import show_sidebar

# Set Streamlit page configuration for wide layout
st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned/cleaned_standardized_phone_usage.csv")

@st.cache_data
def load_model():
    return joblib.load("models/best_classification_model.pkl")

def preprocess_input(input_data):
    """Convert categorical features to numeric encoding"""
    categorical_cols = ["Gender", "Location", "Phone Brand", "OS"]
    
    for col in categorical_cols:
        input_data[col] = pd.factorize(input_data[col])[0]  # Assign unique integer values
    
    return input_data

# Define reverse mapping from numbers to category names
primary_use_mapping = {0: "Education", 1: "Gaming", 2: "Entertainment", 3: "Social Media", 4: "Work"}

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
        prediction = model.predict(input_df)[0]  # Get predicted class (number)
        predicted_category = primary_use_mapping.get(prediction, "Unknown")  # Get category name
        
        st.success(f"📌 **Predicted Primary Use:** {predicted_category}")
        st.session_state["predict"] = False  # Reset state

    # Load data
    data = load_data()


    st.write("### 📊 Data Preview")
    st.dataframe(data.head())  # Display first 5 rows of the data

    display_all_plots(data)  # Display all plots

if __name__ == "__main__":
    main()
