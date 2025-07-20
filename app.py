import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# Config
USE_POLYGON_LIVE = True
CSV_DATA_PATH = "/home/btheard/sentinel/data/all_stocks_5yr.csv"

# Load CSV Data
@st.cache_data
def load_csv_data():
    df = pd.read_csv(CSV_DATA_PATH)
    df.columns = [col.lower().strip() for col in df.columns]
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

# Polygon.io Live Fetch
def fetch_polygon_data(ticker, start, end):
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/1/day/"
        f"{start}/{end}?adjusted=true&sort=asc&limit=5000&apiKey={POLYGON_API_KEY}"
    )
    r = requests.get(url)
    if r.status_code != 200:
        return None
    data = r.json().get('results', [])
    if not data:
        return None
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df['close'] = df['c']
    return df[['date', 'close']].copy()

# UI
st.title("📊 PreMarket Sentinel")

st.sidebar.header("Select Ticker")
default_ticker = "AAPL"
ticker = st.sidebar.text_input("Ticker", value=default_ticker).upper()

st.sidebar.header("Select Date Range")
today = datetime.now()
default_range = [today - timedelta(days=60), today]
date_range = st.sidebar.date_input("Date Range", default_range)

# Dates
if len(date_range) != 2:
    st.warning("Select a valid start and end date.")
    st.stop()

start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

# Load local CSV
csv_df = load_csv_data()

# Use live data if selected range is outside CSV data
use_live = (
    USE_POLYGON_LIVE and
    (start_date > csv_df['date'].max() or end_date > csv_df['date'].max())
)

if use_live:
    st.info("Using live data from Polygon.io")
    df_filtered = fetch_polygon_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    if df_filtered is None or df_filtered.empty:
        st.error("No live data available for this range.")
        st.stop()
else:
    df_filtered = csv_df[
        (csv_df['name'] == ticker) &
        (csv_df['date'] >= start_date) &
        (csv_df['date'] <= end_date)
    ].copy()
    if df_filtered.empty:
        st.error("No historical data for selected range/ticker.")
        st.stop()

# KPI
avg_close = round(df_filtered['close'].mean(), 2)
volatility = round(df_filtered['close'].std() / df_filtered['close'].mean(), 4)
min_close = df_filtered['close'].min()
max_close = df_filtered['close'].max()
last_close = df_filtered.iloc[-1]['close']

st.subheader("📈 KPI Metrics")
st.metric("Average Close", f"${avg_close}")
st.metric("Volatility", f"{volatility*100:.2f}%")
st.metric("Last Close", f"${last_close:.2f}")

# Trade Signal
st.subheader("💡 Trade Recommendation")
signal = "BUY" if last_close < avg_close else "SELL"
st.success(
    f"Based on the selected range, the current price of ${last_close:.2f} suggests: "
    f"{'🟢 BUY - Undervalued' if signal == 'BUY' else '🔴 SELL - Overvalued'}"
)

# Charts
st.subheader("📉 Price Chart")
st.line_chart(df_filtered.set_index('date')['close'])

st.subheader("📊 Volatility")
st.line_chart(df_filtered.set_index('date')['close'].pct_change().rolling(5).std())









