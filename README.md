# Sentinel Forecaster — Options Sweep Interpreter

**Live App:** https://sentinel-baseline-panel.victorioussand-a57f0952.centralus.azurecontainerapps.io/

## What Sentinel is
Sentinel is an **options sweep interpreter**: it scores a historical sweep on:
1) **Directional Bias** (probability the next move is up)  
2) **Volatility Context** (normal vs high-vol regime)  
3) **Expected Next-Day Move** (rough magnitude estimate)

## What Sentinel is not
It is **not** a trading bot and does **not** predict exact prices. It’s designed to **rank and contextualize** sweeps.

## The Problem Sentinel Solves

Options sweeps are widely watched but poorly interpreted.
Most traders rely on heuristics, social feeds, or raw volume signals, which:
- Overweight noise
- Ignore volatility regime
- Fail to rank opportunities systematically

Sentinel addresses this by treating sweep interpretation as a **probabilistic ranking problem**, not a prediction problem.

## Quickstart (60 seconds)
1) Open the app and select a **Historical Options Sweep** (left sidebar)  
2) Read the 3 KPIs at the top  
3) Optionally run **What-If** to test sensitivity (spread/flow changes)  
4) Use the **AI interpretation** as the plain-English takeaway (if enabled)

## What is an Options Sweep?
An **options sweep** is a single order that gets executed across **multiple venues** in rapid succession, usually to get filled quickly at the best available prices.

In practice, sweeps are often interpreted as:
- **Aggressive positioning** (someone wants in *now*)
- **Directional intent** more than passive hedging
- Potential “information events” — but not always “smart money”

Each row in Sentinel’s dataset represents **one historical sweep**, enriched with:
- Price context (spot, strike, spreads)
- Flow intensity (volume / open interest dynamics)
- Volatility-aware engineered features
- Forward-looking labels (next-day direction, regime, return)

Sentinel does **not** assume sweeps are always predictive.  
Instead, it evaluates:
> When sweeps historically mattered — and when they didn’t.

## Data
Sentinel uses a historical engineered sweep dataset:
- File: `data/processed/tradyflow_training.parquet`
- Each row = one historical “sweep” with engineered flow/price features and next-day labels

## Models (Baseline v1)
Sentinel trains and serves three baseline models:
1) **Direction model** → `P(Direction Up)`
2) **Volatility regime model** → `Normal vs High Vol`
3) **Return regression head** → predicted `next_return_1d`

Artifacts live in `models/` and are loaded by the Streamlit app.

## How to read the outputs
- **Directional Bias (Probability):** ~50% = weak/no edge, 55%+ = mild tilt, 60%+ = stronger signal (still not guaranteed)
- **Volatility Context:** high vol = wider swings and harder risk control
- **Expected Next-Day Move:** use for **ranking** expected movement size (not precision)

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

- Short-horizon returns are noisy; predictions are probabilistic, not certain

- Performance depends on dataset coverage and feature stability

- Live sweep ingestion is not wired in this baseline release

## Testing & Validation

Sentinel includes a lightweight test suite to ensure scoring logic is stable,
inputs are validated, and data pipelines behave as expected.

Tests cover:
- Input validation (invalid tickers, malformed data)
- Data pipeline integrity
- Model scoring consistency
- Application wiring and outputs

The goal is not exhaustive testing, but early detection of failures
that would invalidate sweep interpretation.

> For example, test_models.py ensures score stability given identical sweep inputs.