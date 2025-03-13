import streamlit as st
import pandas as pd
import joblib
from plots_main import display_all_plots
from sidebar import show_sidebar

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

primary_use_mapping = {0: "Education", 1: "Gaming", 2: "Entertainment", 3: "Social Media", 4: "Work"}

def main():
    st.title("📱 Phone Usage Prediction & Analysis")

    # Hide sidebar if active tab is not "Main"
    if "active_tab" in st.session_state and st.session_state["active_tab"] != "Main":
        st.markdown("""
            <style>
            [data-testid="stSidebar"] {display: none;}
            </style>
            """, unsafe_allow_html=True)
    else:
        user_input = show_sidebar()  # Sidebar only in "Main"

        st.write("## 🔮 Prediction")
        input_df = pd.DataFrame([user_input])
        input_df = preprocess_input(input_df)

        if "predict" in st.session_state and st.session_state["predict"]:
            model = load_model()
            prediction = model.predict(input_df)[0]
            predicted_category = primary_use_mapping.get(prediction, "Unknown")
            st.success(f"📌 **Predicted Primary Use:** {predicted_category}")
            st.session_state["predict"] = False

    # Load data
    data = load_data()

    st.write("### 📊 Data Preview")
    st.dataframe(data.head())

    display_all_plots(data)

if __name__ == "__main__":
    main()
