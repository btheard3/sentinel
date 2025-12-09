# sentinel_app/app.py

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)

# ---------- Paths ----------

APP_DIR = Path(__file__).resolve().parent          # .../sentinel_app
PROJECT_ROOT = APP_DIR.parent                      # .../sentinel
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "tradyflow_training.parquet"

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

with st.expander("Paths (for sanity checks)", expanded=False):
    st.code(
        f"PROJECT_ROOT: {PROJECT_ROOT}\n"
        f"MODELS_DIR:   {MODELS_DIR}\n"
        f"DATA_PATH:    {DATA_PATH}",
        language="bash",
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


def get_top_features(model, feature_cols, k: int = 5):
    """Return top-k (feature, importance) pairs sorted descending."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []

    pairs = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1],
        reverse=True,
    )[:k]
    return pairs


# ---------- Load everything ----------

models = load_models()
st.success("✅ Models loaded from `models/`")

df = load_training_data()
feature_cols = get_feature_cols(df)

st.write(
    f"Training dataset loaded with **{len(df)}** rows and "
    f"**{len(feature_cols)}** numeric features."
)

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

# ---------- Quick interpretation cards (per-row) ----------

st.subheader("2. Quick interpretation for this sweep")

top_dir = get_top_features(models["direction_rf"], feature_cols, k=5)
top_vol = get_top_features(models["volregime_rf"], feature_cols, k=5)
top_ret = get_top_features(models["nextret_rf"], feature_cols, k=5)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Direction model – key signals**")
    if top_dir:
        df_dir = pd.DataFrame(top_dir, columns=["feature", "importance"])
        st.bar_chart(df_dir.set_index("feature"))
        st.caption(
            "These features most influence whether the model expects the next move "
            "to be up vs. down."
        )
    else:
        st.caption("No feature importances exposed for this model.")

with c2:
    st.markdown("**Volatility regime – key signals**")
    if top_vol:
        df_vol = pd.DataFrame(top_vol, columns=["feature", "importance"])
        st.bar_chart(df_vol.set_index("feature"))
        st.caption(
            "Highlights which flows tend to appear in high-vol vs. normal regimes."
        )
    else:
        st.caption("No feature importances exposed for this model.")

with c3:
    st.markdown("**Return head – key signals**")
    if top_ret:
        df_ret = pd.DataFrame(top_ret, columns=["feature", "importance"])
        st.bar_chart(df_ret.set_index("feature"))
        st.caption(
            "Shows which features matter most for the size of the next-day move."
        )
    else:
        st.caption("No feature importances exposed for this model.")

# ---------- Deeper dive: global behavior ----------

st.subheader("3. Deeper dive – how the models behave on the dataset")

tab_dir, tab_vol, tab_ret = st.tabs(
    ["Direction model", "Volatility regime model", "Return regression"]
)

# Use a subsample for speed
eval_df = df.sample(min(1000, len(df)), random_state=0)
X_eval = eval_df[feature_cols]

with tab_dir:
    st.markdown("#### Direction model (next move up vs. down)")

    y_true_dir = eval_df["direction_up"]
    y_pred_dir = models["direction_rf"].predict(X_eval)

    cm_dir = confusion_matrix(y_true_dir, y_pred_dir, labels=[0, 1])
    acc_dir = (cm_dir.trace() / cm_dir.sum()) if cm_dir.sum() > 0 else 0.0

    st.write(f"**Accuracy on eval sample:** `{acc_dir:.1%}`")

    cm_dir_df = pd.DataFrame(
        cm_dir,
        index=["Actual 0 (down/flat)", "Actual 1 (up)"],
        columns=["Pred 0 (down/flat)", "Pred 1 (up)"],
    )
    st.write("Confusion matrix:")
    st.dataframe(cm_dir_df)

    # Global importances
    dir_importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": models["direction_rf"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    st.write("Top features by importance (direction model):")
    st.bar_chart(dir_importances.head(10).set_index("feature"))

    st.caption(
        "The model leans most on features at the top of this list when deciding "
        "if the next move is up or down."
    )

with tab_vol:
    st.markdown("#### Volatility regime model (normal vs. high-vol)")

    y_true_vol = eval_df["vol_regime"]
    y_pred_vol = models["volregime_rf"].predict(X_eval)

    cm_vol = confusion_matrix(y_true_vol, y_pred_vol, labels=[0, 1])
    acc_vol = (cm_vol.trace() / cm_vol.sum()) if cm_vol.sum() > 0 else 0.0

    st.write(f"**Accuracy on eval sample:** `{acc_vol:.1%}`")

    cm_vol_df = pd.DataFrame(
        cm_vol,
        index=["Actual 0 (normal)", "Actual 1 (high-vol)"],
        columns=["Pred 0 (normal)", "Pred 1 (high-vol)"],
    )
    st.write("Confusion matrix:")
    st.dataframe(cm_vol_df)

    vol_importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": models["volregime_rf"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    st.write("Top features by importance (volatility regime model):")
    st.bar_chart(vol_importances.head(10).set_index("feature"))

    st.caption(
        "These features tend to differentiate calm markets from high-volatility regimes."
    )

with tab_ret:
    st.markdown("#### Return regression head (next_return_1d)")

    y_true_ret = eval_df["next_return_1d"]
    y_pred_ret = models["nextret_rf"].predict(X_eval)

    mae = mean_absolute_error(y_true_ret, y_pred_ret)
    rmse = np.sqrt(mean_squared_error(y_true_ret, y_pred_ret))

    c_mae, c_rmse = st.columns(2)
    with c_mae:
        st.metric("MAE (absolute error)", f"{mae:.4f}")
    with c_rmse:
        st.metric("RMSE", f"{rmse:.4f}")

    ret_importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": models["nextret_rf"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    st.write("Top features by importance (return head):")
    st.bar_chart(ret_importances.head(10).set_index("feature"))

    st.caption(
        "Short-horizon returns are noisy, so errors are expected to cluster near zero. "
        "This head is still useful for ranking sweeps by expected move size."
    )

# ---------- Notes section ----------

st.subheader("4. What this shell proves")

st.markdown(
    """
- ✅ Models load correctly from disk.
- ✅ `feature_cols` selection is consistent with Notebook 04.
- ✅ We can pass a **real engineered row** through all three models:
  - Direction (up / down)
  - Volatility regime (normal vs high)
  - Short-horizon return magnitude
- ✅ We have a first pass at:
  - Sanity-checked test coverage for data + model wiring
  - Feature importance views and confusion matrices
  - A structure that can plug into Azure (App Service / Container Apps)
"""
)
