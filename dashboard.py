import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="LunarAlign Matcher Atlas", layout="wide")

st.title("🌕 LunarAlign Matcher Atlas")
st.markdown("Evaluating computer vision matchers across dynamically simulated lunar illumination.")

atlas_path = "results/atlas.csv"

if not os.path.exists(atlas_path):
    st.warning("Atlas data not found. Please run the sweep script to generate `results/atlas.csv`.")
    st.stop()

df = pd.read_csv(atlas_path)

st.sidebar.header("Filters")
selected_matcher = st.sidebar.selectbox("Select Matcher", df['matcher'].unique())
selected_protocol = st.sidebar.selectbox("Select Protocol", df['protocol'].unique())
selected_factor = st.sidebar.selectbox("Select Factor", df['factor'].unique())

filtered_df = df[
    (df['matcher'] == selected_matcher) &
    (df['protocol'] == selected_protocol) &
    (df['factor'] == selected_factor)
]

if filtered_df.empty:
    st.warning("No data for these filters.")
else:
    # Create pivot tables for heatmaps
    # 1. Success Rate
    success_pivot = filtered_df.pivot_table(index='el', columns='az', values='success', aggfunc='mean') * 100
    
    # 2. Correct Matches
    correct_pivot = filtered_df.pivot_table(index='el', columns='az', values='correct', aggfunc='mean')
    
    st.subheader(f"Success Rate (%) - {selected_matcher.upper()}")
    fig_success = px.imshow(success_pivot, text_auto=True, color_continuous_scale='RdYlGn', origin='lower')
    fig_success.update_layout(xaxis_title="Sun Azimuth (deg)", yaxis_title="Sun Elevation (deg)")
    st.plotly_chart(fig_success, use_container_width=True)

    st.subheader(f"Correct Matches (Inliers) - {selected_matcher.upper()}")
    fig_correct = px.imshow(correct_pivot, text_auto=True, color_continuous_scale='Viridis', origin='lower')
    fig_correct.update_layout(xaxis_title="Sun Azimuth (deg)", yaxis_title="Sun Elevation (deg)")
    st.plotly_chart(fig_correct, use_container_width=True)

st.divider()
st.subheader("Raw Data")
st.dataframe(filtered_df)
