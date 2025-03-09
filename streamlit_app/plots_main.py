import streamlit as st
from plot_utils import display_correlation_heatmap, display_scatter_plot, display_line_plot, display_count_plot, display_age_distribution, display_histogram

def display_all_plots(df):
    st.write("## 📈 Data Insights")

    # Display the Correlation Heatmap outside the row layout (first row only)
    display_correlation_heatmap(df)

    # Second row with 2 plots
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        display_scatter_plot(df)
    with row2_col2:
        display_line_plot(df)

    # Third row with 2 plots
    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        display_count_plot(df)
    with row3_col2:
        display_histogram(df)

    # Fourth row with 2 plots
    row4_col1, row4_col2 = st.columns(2)
    with row4_col1:
        display_age_distribution(df)
    with row4_col2:
        display_scatter_plot(df)
