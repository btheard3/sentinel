# sentinel_app/app.py

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import numpy as np
import pandas as pd

# ---------- Paths ----------
APP_DIR = Path(__file__).resolve().parent          # .../sentinel_app
PROJECT_ROOT = APP_DIR.parent                      # .../sentinel
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "tradyflow_training.parquet"

# Debug: show path
st.write("PROJECT_ROOT:", PROJECT_ROOT)
st.write("MODELS_DIR:", MODELS_DIR)
st.write("DATA_PATH:", DATA_PATH)

# ---------- Streamlit setup ----------

st.set_page_config(
    page_title="Sentinel – Baseline ML Panel",
    layout="wide",
)

st.title("🔮 Sentinel – Baseline Modeling Panel")
st.caption(
    "Baseline ML models trained on options sweep features. "
    "This shell wires notebooks → models → interactive dashboard."
)


# ---------- Load artifacts ----------

@st.cache_resource
def load_models():
    """Load trained models from the models/ directory."""
    direction_rf = joblib.load(MODELS_DIR / "sentinel_direction_up_rf.pkl")
    volregime_rf = joblib.load(MODELS_DIR / "sentinel_vol_regime_rf.pkl")
    nextret_rf = joblib.load(MODELS_DIR / "sentinel_next_return_rf.pkl")

    return {
        "direction_rf": direction_rf,
        "volregime_rf": volregime_rf,
        "nextret_rf": nextret_rf,
    }

@st.cache_data
def load_training_data():
    """Load the modeling dataset with features + targets."""
    df = pd.read_parquet(DATA_PATH)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Recreate the feature column selection from Notebook 04.

    IMPORTANT:
    This MUST match the logic used when training the models
    (same columns, same order). If it ever drifts, copy the exact
    block from 04_model_training.ipynb.
    """
    # Columns we do NOT want as features
    exclude_cols = {
        "Time",
        "Sym",
        "C/P",
        "Exp",
        "next_spot",
        "next_return_1d",
        "direction_up",
        "vol_regime",
    }

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    return feature_cols


# ---------- Load everything ----------

models = load_models()
st.success("Models loaded from `models/`")

df = load_training_data()
feature_cols = get_feature_cols(df)

st.write(f"Training dataset loaded with **{len(df)}** rows and **{len(feature_cols)}** features.")


# ---------- Sidebar: pick a sample sweep ----------

with st.sidebar:
    st.header("Sample Sweep Selector")
    st.caption("Use a real row from the training set to sanity-check model wiring.")

    max_idx = len(df) - 1
    idx = st.number_input(
        "Row index",
        min_value=0,
        max_value=max_idx,
        value=0,
        step=1,
    )

    sample = df.iloc[[idx]]  # keep as DataFrame


# ---------- Show input features ----------

st.subheader("1. Input Features (Engineered Row)")

st.dataframe(sample[feature_cols], use_container_width=True)


# ---------- Run predictions ----------

X_sample = sample[feature_cols]

# Direction: probability next move is up
dir_proba_up = models["direction_rf"].predict_proba(X_sample)[0, 1]
dir_label = "⬆️ Up" if dir_proba_up >= 0.5 else "⬇️ Down / Flat"

# Vol regime: 0 = normal, 1 = high vol
vol_pred = models["volregime_rf"].predict(X_sample)[0]
vol_proba = models["volregime_rf"].predict_proba(X_sample)[0, int(vol_pred)]
vol_label = "🌪 High volatility" if vol_pred == 1 else "🌤 Normal volatility"

# Next 1D return prediction
next_ret_pred = models["nextret_rf"].predict(X_sample)[0]


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="P(Direction Up)",
        value=f"{dir_proba_up:.1%}",
        delta=dir_label,
    )

with col2:
    st.metric(
        label="Volatility Regime",
        value=vol_label,
        delta=f"Confidence {vol_proba:.1%}",
    )

with col3:
    st.metric(
        label="Predicted Next 1D Return",
        value=f"{next_ret_pred:.3f}",
    )


# ---------- Notes section ----------

st.subheader("2. What this shell proves")

st.markdown(
    """
- ✅ Models load correctly from disk.
- ✅ `feature_cols` selection is consistent with Notebook 04.
- ✅ We can pass a **real engineered row** through all three models:
  - Direction (up / down)
  - Volatility regime (normal vs high)
  - Short-horizon return magnitude
- ✅ This structure is ready for:
  - A user-friendly input form for new sweeps
  - Feature importance charts and confusion matrices
  - Integration into Azure (App Service / Container Apps)
"""
)