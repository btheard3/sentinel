# Sentinel Forecaster

Interpreting Options Sweeps Under Uncertainty

Live App:
https://sentinel-baseline-panel.victorioussand-a57f0952.centralus.azurecontainerapps.io/

## The Problem

Options sweeps are widely monitored but routinely misinterpreted.

Large, fast executions are often assumed to represent “smart money” or strong directional intent.
In practice, many sweeps are hedges, volatility plays, or statistically unremarkable without context.

Relying on sweep size or social heuristics creates false certainty and poor risk decisions.

---

## Why This Matters

The challenge is not detecting sweeps — it is knowing **when a sweep historically mattered**.

Most retail tools surface activity, not context.
Sentinel was built to provide probabilistic interpretation, not trade signals.

---

## What Was Built

Sentinel evaluates a *single historical options* sweep using three independent lenses:

1. **Directional Bias**
Probability the underlying moves up the next trading day

2. **Volatility Context**
Classification of the market environment (Normal vs High Vol)

3. **Expected Next-Day Move**
Rough magnitude estimate used for ranking, not precision

The system is intentionally conservative and interpretability-first.

---

## Data

- Public options sweep dataset (TradyFlow via Kaggle)

- File: `data/processed/tradyflow_training.parquet`

- Each row represents one historical sweep with:

    - Price & flow features

    - Volatility-aware engineered inputs

    - Next-day direction, regime, and return labels

Fully local and reproducible.

---

## Modeling

Sentinel uses three baseline models:

- Direction classifier → `P(Direction Up)`

- Volatility regime classifier → Normal vs High Vol

- Return regression head → predicted next-day return magnitude

Models are trained offline and loaded by the Streamlit app at runtime.

---

## What Happened

Key observations from analysis:

- Most sweeps cluster near ~50% directional probability

- Volatility regime materially alters outcome distributions

- Magnitude estimates are more useful for ranking than forecasting

- Large sweeps alone are weak signals without context

In many cases, Sentinel indicated **low informational value**, helping avoid over-trading.

---

## Limitations

- Short-horizon returns are highly noisy

- No execution or PnL modeling

- No live sweep ingestion in this baseline

- Performance depends on regime stability and dataset coverage

This is a **decision-support tool**, not a trading system.

---

## Future Work

Planned next steps include:

- Live sweep ingestion and streaming updates

- Regime-aware retraining and drift monitoring

- Strategy-specific overlays (e.g., earnings, index vs single-name)

- Portfolio-level aggregation across multiple sweeps

- Extended horizon labels beyond next-day returns

These extensions build on the same interpretability-first foundation.

---

## Reproducibility
```bash
pip install -r requirements.txt
streamlit run sentinel_app/app.py
```


Docker:
```bash
docker build -t sentinel .
docker run -p 8501:8501 sentinel
```
---

## What This Demonstrates

- Applied ML in a noisy financial domain

- Volatility-aware modeling

- Probabilistic reasoning over heuristics

- Production deployment (Streamlit + Azure)

- Clear separation of modeling and interpretation

---

Author: Brandon Theard
Data Scientist | Decision Support Systems