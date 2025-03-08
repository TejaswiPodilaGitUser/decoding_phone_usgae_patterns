import streamlit as st

def show_sidebar():
    # Sidebar for feature selection
    st.sidebar.title("🔍 User Input Features")
    
    # Sidebar input fields for each column
    age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=30)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    location = st.sidebar.selectbox("Location", ["Mumbai", "Delhi", "Pune", "Ahmedabad"])
    phone_brand = st.sidebar.selectbox("Phone Brand", ["Vivo", "Realme", "Nokia", "Samsung", "Xiaomi"])
    os = st.sidebar.selectbox("OS", ["Android", "iOS"])
    screen_time = st.sidebar.slider("Screen Time (hrs/day)", 0.0, 24.0, 3.0)
    data_usage = st.sidebar.slider("Data Usage (GB/month)", 0.0, 100.0, 5.0)
    calls_duration = st.sidebar.slider("Calls Duration (mins/day)", 0, 500, 30)
    num_apps_installed = st.sidebar.slider("Number of Apps Installed", 0, 100, 20)
    social_media_time = st.sidebar.slider("Social Media Time (hrs/day)", 0.0, 10.0, 1.0)
    ecommerce_spend = st.sidebar.slider("E-commerce Spend (INR/month)", 0, 5000, 1000)
    streaming_time = st.sidebar.slider("Streaming Time (hrs/day)", 0.0, 10.0, 1.0)
    gaming_time = st.sidebar.slider("Gaming Time (hrs/day)", 0.0, 10.0, 1.0)
    monthly_recharge_cost = st.sidebar.slider("Monthly Recharge Cost (INR)", 0, 1000, 300)
    battery_consumption = st.sidebar.slider("Battery Consumption (mAh/day)", 0, 5000, 2000)

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
