import pandas as pd
import yfinance as yf
import numpy as np

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_indicators(tickers):
    results = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="15d", interval="1d")

            if hist.empty or len(hist) < 6:
                continue

            hist['rsi'] = compute_rsi(hist['Close'])
            rsi_latest = hist['rsi'].iloc[-1]

            rel_volume = hist['Volume'].iloc[-1] / hist['Volume'].rolling(window=5).mean().iloc[-1]
            close_price = hist['Close'].iloc[-1]
            high_5d = hist['High'].rolling(window=5).max().iloc[-1]
            low_5d = hist['Low'].rolling(window=5).min().iloc[-1]
            dist_from_high = (close_price - high_5d) / high_5d * 100
            dist_from_low = (close_price - low_5d) / low_5d * 100

            results.append({
                'ticker': ticker,
                'rsi': round(rsi_latest, 2),
                'rel_volume': round(rel_volume, 2),
                'dist_from_5d_high': round(dist_from_high, 2),
                'dist_from_5d_low': round(dist_from_low, 2),
            })

        except Exception as e:
            print(f"Error computing indicators for {ticker}: {e}")

    return pd.DataFrame(results)
