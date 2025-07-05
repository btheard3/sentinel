import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="📈 PreMarket Sentinel – AI-Powered Trade Scanner", layout="wide")

st.title("📈 PreMarket Sentinel – AI-Powered Trade Scanner")

# Path to saved report
report_path = "notebooks/daily_report.csv"

# Try to load saved report
if os.path.exists(report_path):
    try:
        df = pd.read_csv(report_path)

        if not df.empty:
            st.success("✅ Report loaded successfully.")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("⚠️ Report is empty. Please check your data.")
    except Exception as e:
        st.error(f"❌ Failed to load report: {e}")
else:
    st.warning("⚠️ No saved report found. Please run the notebook or script to generate `daily_report.csv`.")

