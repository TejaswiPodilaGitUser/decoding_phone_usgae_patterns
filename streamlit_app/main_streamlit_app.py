import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eda.eda import main as eda_page
from eda.eda_processed import main as eda_processed_page
from streamlit_app.clustering_page import main as clustering_page
from streamlit_app.main import main as main_page

# Initialize session state for active tab
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "EDA"  # Default tab

# Define Tabs
tab1, tab2, tab3, tab4 = st.tabs(["EDA", "EDA Processed", "Clustering", "Main"])

# Switch tabs and update session state
with tab1:
    st.session_state["active_tab"] = "EDA"
    eda_page()

with tab2:
    st.session_state["active_tab"] = "EDA Processed"
    eda_processed_page()

with tab3:
    st.session_state["active_tab"] = "Clustering"
    clustering_page()

with tab4:
    st.session_state["active_tab"] = "Main"
    main_page()
