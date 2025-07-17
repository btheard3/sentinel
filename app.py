import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Load Data
df = pd.read_csv("data/all_stocks_5yr.csv")
df.columns = df.columns.str.lower()
df['date'] = pd.to_datetime(df['date'])

# Streamlit Sidebar Filters
tickers = df['name'].unique().tolist()
selected_ticker = st.sidebar.selectbox("Select Ticker", tickers)

min_date = df['date'].min()
max_date = df['date'].max()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Main Title
st.title("📈 PreMarket Sentinel")
st.caption("Analyze historical stock signals and model predictions.")

# Filter Data Safely
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (
        (df['name'] == selected_ticker) &
        (df['date'] >= pd.to_datetime(start_date)) &
        (df['date'] <= pd.to_datetime(end_date))
    )
    filtered = df[mask]
else:
    st.warning("Please select a valid date range.")
    st.stop()

# Display Chart
st.subheader(f"📊 Price Trend for {selected_ticker}")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(filtered['date'], filtered['close'], label='Close Price', color='skyblue')
ax.set_title(f"{selected_ticker} Close Price Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

st.markdown("### 📌 Key Stats")
col1, col2, col3 = st.columns(3)

col1.metric("📅 Date Range", f"{start_date} → {end_date}")
col2.metric("📊 Avg Close", f"${filtered['close'].mean():.2f}")
col3.metric("📈 Max Close", f"${filtered['close'].max():.2f}")

# Volume Trend
st.subheader(f"📦 Volume Trend for {selected_ticker}")
fig_vol, ax_vol = plt.subplots(figsize=(12, 3))
ax_vol.bar(filtered['date'], filtered['volume'], color='orange')
ax_vol.set_ylabel("Volume")
st.pyplot(fig_vol)

# Rolling Volatility
st.subheader("📉 20-Day Rolling Volatility")
filtered['volatility'] = filtered['close'].pct_change().rolling(window=20).std()
fig_vol, ax_vol = plt.subplots(figsize=(12, 3))
ax_vol.plot(filtered['date'], filtered['volatility'], color='red')
ax_vol.set_ylabel("Volatility")
st.pyplot(fig_vol)

st.markdown("### 🧠 Summary")
st.info(f"""
From {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}, 
**{selected_ticker}** traded between **${filtered['close'].min():.2f}** and **${filtered['close'].max():.2f}**.
The average close price was **${filtered['close'].mean():.2f}**, with trading volume peaking at **{filtered['volume'].max():,}**.
""")




