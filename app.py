# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from openai import OpenAIError
import openai
import os

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/all_stocks_5yr.csv")
    df.columns = [col.lower().strip() for col in df.columns]
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

df = load_data()

# Sidebar Filters
tickers = df['name'].unique().tolist()
selected_ticker = st.sidebar.selectbox("Select Ticker", tickers)

min_date = df['date'].min()
max_date = df['date'].max()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Main Title
st.title("📈 PreMarket Sentinel")
st.caption("Analyze historical stock signals, KPIs, and trade recommendations.")

# Filter Data
mask = (
    (df['name'] == selected_ticker) &
    (df['date'] >= pd.to_datetime(date_range[0])) &
    (df['date'] <= pd.to_datetime(date_range[1]))
)
filtered = df[mask]

# Price Chart
st.subheader(f"📉 Price Trend for {selected_ticker}")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(filtered['date'], filtered['close'], label='Close Price', color='skyblue')
ax.set_title(f"{selected_ticker} Close Price Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# KPIs
avg_close = round(filtered['close'].mean(), 2)
max_close = round(filtered['close'].max(), 2)
min_close = round(filtered['close'].min(), 2)

st.markdown("### 🌟 Key Stats")
col1, col2, col3, col4 = st.columns(4)
col1.metric("🗓️ Date Range", f"{date_range[0]} → {date_range[1]}")
col2.metric("📊 Avg Close", f"${avg_close}")
col3.metric("📈 Max Close", f"${max_close}")
col4.metric("📉 Min Close", f"${min_close}")

# Volume Chart
st.subheader(f"📦 Volume Trend for {selected_ticker}")
fig, ax = plt.subplots(figsize=(10, 2))
ax.bar(filtered['date'], filtered['volume'], color='orange', width=2)
ax.set_title("Daily Trading Volume")
ax.set_ylabel("Volume")
st.pyplot(fig)

# Rolling Volatility
st.subheader("📉 20-Day Rolling Volatility")
filtered['volatility'] = filtered['close'].pct_change().rolling(window=20).std()
fig, ax = plt.subplots(figsize=(10, 2))
ax.plot(filtered['date'], filtered['volatility'], color='red')
ax.set_ylabel("Volatility")
st.pyplot(fig)

# Trade Recommendation
last_close = filtered['close'].iloc[-1]
threshold_buy = avg_close * 0.90
threshold_sell = avg_close * 1.10

if last_close < threshold_buy:
    trade_signal = "🟢 BUY - Undervalued"
elif last_close > threshold_sell:
    trade_signal = "🔴 SELL - Overvalued"
else:
    trade_signal = "🟡 HOLD - Fairly Valued"

st.markdown("### 💡 Trade Recommendation")
st.success(f"Based on the selected date range, the current close price of **${last_close:.2f}** suggests: **{trade_signal}**")

# Summary with optional LLM
st.subheader("🧠 Summary")
default_summary = f"""
From {date_range[0]} to {date_range[1]}, {selected_ticker} traded between ${min_close} and ${max_close}.
The average close price was ${avg_close}, with trading volume peaking at {int(filtered['volume'].max()):,}.
"""

if "OPENAI_API_KEY" in os.environ:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    try:
        prompt = f"Summarize this stock data insight: {default_summary}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        st.info(response.choices[0].message.content.strip())
    except OpenAIError as e:
        st.warning(f"LLM summary failed: {e}")
        st.text(default_summary)
else:
    st.text(default_summary)


