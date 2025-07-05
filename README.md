## 📘 What Is a Gap?

A **gap** happens when a stock opens significantly higher or lower than its previous day's closing price. Gaps often indicate strong momentum, major news, or upcoming volatility — perfect conditions for options trades.

### 📈 Gap Percentage (`gap_pct`)

We calculate the **gap %** like this:

gap_pct = ((open - prev_close) / prev_close) \* 100

- **Gap Up** → The stock opens above the previous close (bullish sign)
- **Gap Down** → The stock opens below the previous close (bearish signal)

Example:

- If TSLA closed at $250 and opens at $260:
  - `gap_pct = ((260 - 250) / 250) * 100 = 4%`
  - That’s a **4% gap up**
