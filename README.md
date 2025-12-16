# Sentinel Forecaster — Options Sweep Interpreter

**Live App:** <https://sentinel-baseline-panel.victorioussand-a57f0952.centralus.azurecontainerapps.io/>

## What Sentinel is
Sentinel is a **sweep interpreter**: it scores an options sweep on (1) directional bias, (2) volatility regime context, and (3) expected next-day move.

## What Sentinel is not
It is **not** a trading bot and does **not** predict exact prices. It’s designed to **rank and contextualize** sweeps.

## What is an Options Sweep?

An **options sweep** is a large, aggressive options trade that executes across multiple exchanges in rapid succession, typically at the ask (for calls) or bid (for puts).

In practice, sweeps are often interpreted as:
- **Urgent positioning** by institutional or informed traders
- **Directional intent** rather than passive hedging
- A signal that someone is willing to pay up for exposure

Each row in Sentinel’s dataset represents **one historical sweep**, enriched with:
- Price context (spot, strike, spreads)
- Flow intensity (volume, open interest dynamics)
- Volatility-aware features
- Forward-looking labels (next-day direction, regime, return)

Sentinel does **not** assume sweeps are always “smart money.”
Instead, it evaluates:
> *When sweeps historically mattered — and when they didn’t.*

## Data
Sentinel uses a historical engineered sweep dataset:
- File: `data/processed/tradyflow_training.parquet`
- Each row = one historical “sweep” with engineered flow/price features and next-day labels.

## Models (Baseline v1)
Sentinel trains and serves three baseline models:
1) **Direction model** → `P(Direction Up)`
2) **Volatility regime model** → `Normal vs High Vol`
3) **Return regression head** → predicted `next_return_1d`

Artifacts live in `models/` and are loaded by the Streamlit app.

## How to read the outputs
- **Directional Bias (Probability):** near 50% = weak/no edge, 55%+ = mild tilt, 60%+ = stronger signal (still not guaranteed)
- **Volatility Context:** high vol = wider swings and harder risk control
- **Expected Next-Day Move:** used for ranking sweeps by expected movement size

## Run locally
```bash
pip install -r requirements.txt
streamlit run sentinel_app/app.py
```

## Run with Docker
```bash
docker build -t sentinel .
docker run -p 8501:8501 sentinel
```

## Deployment

This app is deployed on Azure Container Apps via GitHub Actions CI/CD.

## Limitations

Short-horizon returns are noisy; predictions are probabilistic, not certain.

Trained on historical engineered features; performance depends on dataset coverage.

Live sweep ingestion is not wired in this baseline release.