import yfinance as yf
import pandas as pd
from datetime import datetime
import os

def fetch_premarket_movers():
    # Placeholder tickers for testing
    tickers = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT']

    data = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2d", interval="1d")

            if len(hist) >= 2:
                prev_close = hist['Close'].iloc[-2]
                latest_open = hist['Open'].iloc[-1]
                change_pct = ((latest_open - prev_close) / prev_close) * 100

                data.append({
                    'ticker': ticker,
                    'prev_close': round(prev_close, 2),
                    'open': round(latest_open, 2),
                    'gap_pct': round(change_pct, 2),
                    'timestamp': datetime.now().isoformat()
                })

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    df = pd.DataFrame(data)
    return df

def save_report(df, path="notebooks/daily_report.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Saved report to {path}")

if __name__ == "__main__":
    df = fetch_premarket_movers()
    if not df.empty:
        save_report(df)
    else:
        print("⚠️ No data fetched.")
