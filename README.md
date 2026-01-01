# Sentinel Forecaster — Interpreting Options Sweeps Under Uncertainty
🔗 Live App

https://sentinel-baseline-panel.victorioussand-a57f0952.centralus.azurecontainerapps.io/

## Problem

Options sweeps are often assumed to represent “smart money” despite weak empirical validation. Large executions without context lead to false certainty.

## Why This Problem Matters

Misinterpreting flow leads to:

- Directional overconfidence

- Volatility blindness

- Fragile decision-making

## Data Used

- **TradyFlow options sweep dataset (Kaggle)**

- Each row represents a historical sweep with engineered features and labeled outcomes

## Approach

- Sweep-level feature engineering

- Probabilistic scoring across:

    - Directional bias

    - Volatility regime

    - Expected impact

- Offline-trained baseline models loaded by a live Streamlit app

This is a **decision-support system**, not a trading bot.

## Evaluation & Findings

- Most sweeps cluster near neutral probabilities

- Volatility regime matters more than strike selection

- Sweep size alone is a weak signal

## Limitations

- No live sweep ingestion

- Short-horizon noise

- No execution or PnL modeling

## Planned Next Steps

- Probability calibration curves

- Structural clustering of sweeps

- Portfolio-level aggregation views

## Reproducibility — Run Locally

Option 1: Streamlit
```bash
git clone https://github.com/btheard3/sentinel
cd sentinel
pip install -r requirements.txt
streamlit run sentinel_app/app.py
```

Option 2: Docker
```bash
docker build -t sentinel .
docker run -p 8501:8501 sentinel
```

## Portfolio Context

**Flow interpretation layer** — converts noisy options activity into probabilistic context.

Author

Brandon Theard
Data Scientist | Decision-Support Systems

GitHub: https://github.com/btheard3

LinkedIn: https://www.linkedin.com/in/brandon-theard-811b38131/