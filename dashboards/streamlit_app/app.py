import streamlit as st
import pandas as pd
from pathlib import Path

from ...src.config import JOURNAL_DIR, PROJECT_NAME
from ...src.journal.excel_writer import DEFAULT_JOURNAL

st.set_page_config(page_title="Sentinel Dashboard", layout="wide")

st.title(PROJECT_NAME)
st.caption("Premarket forecasting, ML signals, and Excel-based trade journaling.")

journal_path = Path(DEFAULT_JOURNAL)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Today’s Signals (placeholder)")
    st.info("No live model yet – this is the initial scaffold for Sentinel.")

with col2:
    st.subheader("Trade Journal Snapshot")
    if journal_path.exists():
        df = pd.read_excel(journal_path)
        st.metric("Total Logged Trades", len(df))
        st.dataframe(df.tail(10), use_container_width=True)
    else:
        st.warning("No journal file found yet. Once we start logging trades, they will appear here.")
