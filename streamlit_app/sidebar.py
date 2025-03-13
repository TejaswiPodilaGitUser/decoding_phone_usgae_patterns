import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned/cleaned_standardized_phone_usage.csv")

def show_sidebar():
    # Load dataset to get unique values for dropdowns
    data = load_data()
    
    # Sidebar for feature selection
    st.sidebar.title("🔍 User Input Features")
    
    # Sidebar input fields with dynamic dropdown values
    age = st.sidebar.number_input("Age", min_value=int(data["Age"].min()), max_value=int(data["Age"].max()), value=30)
    gender = st.sidebar.selectbox("Gender", data["Gender"].unique())
    location = st.sidebar.selectbox("Location", data["Location"].unique())
    
    # Filtering Phone Brand and OS dropdowns dynamically
    phone_brands = data["Phone Brand"].unique()
    os_options = data["OS"].unique()
    
    phone_brand = st.sidebar.selectbox("Phone Brand", phone_brands)
    
    if phone_brand.lower() == "apple":
        os = st.sidebar.selectbox("OS", ["iOS"])
    else:
        os = st.sidebar.selectbox("OS", [o for o in os_options if o.lower() != "ios"])  # Exclude iOS for non-Apple brands
    
    if os.lower() == "ios":
        phone_brand = st.sidebar.selectbox("Phone Brand", ["Apple"])  # Lock to Apple if iOS is selected
    
    screen_time = st.sidebar.slider("Screen Time (hrs/day)", float(data["Screen Time (hrs/day)"].min()), float(data["Screen Time (hrs/day)"].max()), 3.0)
    data_usage = st.sidebar.slider("Data Usage (GB/month)", float(data["Data Usage (GB/month)"].min()), float(data["Data Usage (GB/month)"].max()), 5.0)
    calls_duration = st.sidebar.slider("Calls Duration (mins/day)", int(data["Calls Duration (mins/day)"].min()), int(data["Calls Duration (mins/day)"].max()), 30)
    num_apps_installed = st.sidebar.slider("Number of Apps Installed", int(data["Number of Apps Installed"].min()), int(data["Number of Apps Installed"].max()), 20)
    social_media_time = st.sidebar.slider("Social Media Time (hrs/day)", float(data["Social Media Time (hrs/day)"].min()), float(data["Social Media Time (hrs/day)"].max()), 1.0)
    ecommerce_spend = st.sidebar.slider("E-commerce Spend (INR/month)", int(data["E-commerce Spend (INR/month)"].min()), int(data["E-commerce Spend (INR/month)"].max()), 1000)
    streaming_time = st.sidebar.slider("Streaming Time (hrs/day)", float(data["Streaming Time (hrs/day)"].min()), float(data["Streaming Time (hrs/day)"].max()), 1.0)
    gaming_time = st.sidebar.slider("Gaming Time (hrs/day)", float(data["Gaming Time (hrs/day)"].min()), float(data["Gaming Time (hrs/day)"].max()), 1.0)
    monthly_recharge_cost = st.sidebar.slider("Monthly Recharge Cost (INR)", int(data["Monthly Recharge Cost (INR)"].min()), int(data["Monthly Recharge Cost (INR)"].max()), 300)
    battery_consumption = st.sidebar.slider("Battery Consumption (mAh/day)", int(data["Battery Consumption (mAh/day)"].min()), int(data["Battery Consumption (mAh/day)"].max()), 2000)
    
    # Create a dictionary of all the selected features
    features = {
        "Age": age,
        "Gender": gender,
        "Location": location,
        "Phone Brand": phone_brand,
        "OS": os,
        "Screen Time (hrs/day)": screen_time,
        "Data Usage (GB/month)": data_usage,
        "Calls Duration (mins/day)": calls_duration,
        "Number of Apps Installed": num_apps_installed,
        "Social Media Time (hrs/day)": social_media_time,
        "E-commerce Spend (INR/month)": ecommerce_spend,
        "Streaming Time (hrs/day)": streaming_time,
        "Gaming Time (hrs/day)": gaming_time,
        "Monthly Recharge Cost (INR)": monthly_recharge_cost,
        "Battery Consumption (mAh/day)": battery_consumption
    }
    
    # Prediction Button in Sidebar
    if st.sidebar.button("⚡ Predict Primary Use"):
        st.session_state["predict"] = True  # Store state to trigger prediction
    
    return features
