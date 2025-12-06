# Sentinel – Premarket Forecasting & Trade Journal

**Sentinel** is an AI-powered premarket engine that:
- Scans premarket movers
- Engineers predictive features from options flow and price action
- Uses ML models trained on the TradyFlow options dataset
- Generates continuation / fade signals before the bell
- Logs every idea into an **Excel trade journal**
- Serves a live dashboard (Streamlit) deployed on **Azure**

## High-Level Architecture

1. **Offline training**
   - Load TradyFlow options dataset (Kaggle)
   - Engineer features (gap %, volume surge, flow pressure, etc.)
   - Train gradient-boosted models (XGBoost / LightGBM)
   - Save models to `models/`

2. **Live premarket pipeline**
   - Pull live data from Polygon
   - Build the same feature set
   - Run inference and produce signals
   - Write trades + signals to the Excel journal

3. **Dashboard**
   - Streamlit app in `dashboards/streamlit_app`
   - Shows today’s signals, recent performance, and journal stats
   - Deployed via Docker → Azure App Service

4. **Azure**
   - App Service for dashboard
   - ACR for Docker images
   - Azure Functions for scheduled premarket runs
   - Blob Storage for journal archives and logs

## Repo Layout

```text
sentinel/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/          # Original TradyFlow CSVs, Polygon pulls
│   ├── processed/    # Feature tables for modeling
│   └── external/     # Lookups, metadata, etc.
├── notebooks/
│   ├── 00_sentinel_overview.ipynb
│   ├── 01_tradyflow_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_live_inference_pipeline.ipynb
│   └── 05_dashboard_and_results.ipynb
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── load_tradyflow.py
│   │   └── polygon_live.py
│   ├── features/
│   │   └── feature_builder.py
│   ├── models/
│   │   ├── train.py
│   │   └── infer.py
│   ├── journal/
│   │   └── excel_writer.py
│   └── utils/
│       └── logging_utils.py
├── dashboards/
│   └── streamlit_app/
│       └── app.py
├── trade_journal/
│   └── template/
│       └── sentinel_trade_journal_template.xlsx  # Excel template (to be created)
└── azure_deploy/
    ├── Dockerfile
    └── azure_notes.md

