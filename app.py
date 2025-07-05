import streamlit as st
import pandas as pd

st.set_page_config(page_title="PreMarket Sentinel", layout="wide")

st.title("📈 PreMarket Sentinel – AI-Powered Trade Scanner")

# Load saved daily report CSV or pull from notebook
try:
    df = pd.read_csv("daily_report.csv")
    st.success("Loaded today's Sentinel report.")
except:
    st.warning("No saved report found. Please run the notebook first.")
    st.stop()

# Filter by setup
setup_filter = st.multiselect("Filter by Setup", options=df["Setup"].unique(), default=list(df["Setup"].unique()))
filtered = df[df["Setup"].isin(setup_filter)]

# Show table
st.dataframe(filtered, use_container_width=True)

# Summary stats
st.markdown("### Summary Stats")
st.metric("Tickers Flagged", len(filtered))
st.metric("Avg Premarket Move", f"{filtered['Premarket % Change'].mean():.2f}%")

# Export option
st.download_button("📥 Download Report", data=filtered.to_csv(index=False), file_name="sentinel_report.csv")
