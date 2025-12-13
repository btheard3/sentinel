# sentinel_app/app.py

from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error

load_dotenv()  # loads variables from .env into environment

# ---------- Streamlit setup ----------
st.set_page_config(
    page_title="Sentinel – Baseline ML Panel",
    layout="wide",
)

# ---------- Optional OpenAI ----------
try:
    # New OpenAI SDK
    from openai import OpenAI
except ImportError:
    OpenAI = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SENTINEL_ENV = os.getenv("SENTINEL_ENV", "dev")

# ---------- Paths ----------
APP_DIR = Path(__file__).resolve().parent          # .../sentinel_app
PROJECT_ROOT = APP_DIR.parent                      # .../sentinel
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "tradyflow_training.parquet"

# ---------- Global UI CSS (narrow + readable) ----------
st.markdown(
    """
<style>
/* Constrain main content width */
.block-container {
  max-width: 1100px;
  padding-top: 2rem;
  padding-bottom: 3rem;
}

/* Slightly reduce sidebar dominance on desktop */
section[data-testid="stSidebar"] {
  width: 320px !important;
}

/* Make big dataframes less overwhelming */
[data-testid="stDataFrame"] {
  border-radius: 10px;
}

/* Headings spacing */
h1, h2, h3 { margin-bottom: 0.5rem; }

/* Make metric deltas a bit calmer */
[data-testid="stMetricDelta"] {
  font-size: 0.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Title ----------
st.title("🔮 Sentinel – Baseline Modeling Panel")
st.info(
    """
**How to read Sentinel (60 seconds)**

- Sentinel is a **sweep interpreter**, not a trading bot.
- It scores a sweep on **direction**, **volatility regime**, and **expected next-day move**.
- Use it to **rank** sweeps and understand risk context — not to predict exact prices.
- Start with a **Historical Options Sweep** (left). Then optionally run a **What-If Scenario**.
- The **AI interpretation** summarizes what the model is implying in plain English.
""",
    icon="🧭",
)

st.markdown(
    """
**What this is:** a research-grade “sweep interpreter.”  
Pick a real sweep row (or tweak one manually) and Sentinel returns:

- **P(Direction Up):** probability the next move is up  
- **Volatility Regime:** normal vs high-vol market conditions  
- **Expected 1D Return:** rough size/direction estimate for the next day  

**How to use it (simple):**
1) Pick a sweep row (left sidebar)  
2) Optionally tweak Spot/Strike/Spread/Flow  
3) Read the AI interpretation as the plain-English summary
"""
)

# ---------- Sidebar: mode + env ----------
with st.sidebar:
    st.markdown("## Mode")
    VIEW_MODE = st.radio(
        "Choose what you want to see:",
        ["User view", "Diagnostics (advanced)"],
        index=0,
        label_visibility="collapsed",
    )

    st.caption(f"Environment: `{SENTINEL_ENV}`")

    if OPENAI_API_KEY is None:
        st.warning("OPENAI_API_KEY is not set. AI explanations will be disabled.", icon="⚠️")

# ---------- Helpers ----------
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
    # NOTE: this path must exist inside the container too
    return pd.read_parquet(DATA_PATH)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Recreate feature selection logic from training notebook.

    IMPORTANT:
    This must match the logic used when training the models
    (same columns, same order).
    """
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
    pairs = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:k]
    return pairs


def generate_manual_ai_summary(
    manual_inputs: dict,
    dir_proba_up: float,
    vol_label: str,
    vol_proba: float,
    next_ret_pred: float,
) -> str:
    """
    Use OpenAI to explain what the manual sweep predictions mean
    in plain English for a trader.
    Falls back gracefully if OPENAI_API_KEY or the SDK are missing.
    """
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return (
            "AI interpretation is disabled (missing OpenAI dependency or OPENAI_API_KEY). "
            "You can still use the raw model predictions above."
        )

    client = OpenAI()  # reads OPENAI_API_KEY from env

    prompt = f"""
You are an options quant explaining model output to a trader.

Manual sweep inputs (user-edited):
{manual_inputs}

Model predictions on this manual sweep:
- P(next move up) = {dir_proba_up:.3f}
- Volatility regime = {vol_label} (model confidence {vol_proba:.3f})
- Predicted next 1-day return = {next_ret_pred:.4f}

Write a short, concrete interpretation:
1) What the model is expecting (direction + size of move).
2) How volatility regime influences risk.
3) One or two practical takeaways.

Use 3–5 bullet points. Avoid hype. Be realistic.
""".strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a calm, realistic options strategist. "
                        "Explain predictions clearly, no jargon overload."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=260,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"AI interpretation failed: {exc}"


# ---------- Load models (fast) ----------
models = load_models()

# ============================
# USER VIEW
# ============================
if VIEW_MODE == "User view":

    # Load df only here (needed for sample selector)
    df = load_training_data()
    feature_cols = get_feature_cols(df)

    st.write(
        f"Training dataset loaded with **{len(df)}** rows and "
        f"**{len(feature_cols)}** numeric features."
    )

    # Sidebar: pick a sample sweep
    with st.sidebar:
        st.header("Sample Sweep Selector")
        st.caption("Pick a real row from training data.")

        max_idx = len(df) - 1
        idx = st.number_input(
            "Row index",
            min_value=0,
            max_value=max_idx,
            value=0,
            step=1,
        )

    sample = df.iloc[[idx]]  # keep as DataFrame

    # Predictions on selected sweep
    X_sample = sample[feature_cols]

    dir_proba_up = models["direction_rf"].predict_proba(X_sample)[0, 1]
    dir_label = "⬆️ Up" if dir_proba_up >= 0.5 else "⬇️ Down / Flat"

    vol_pred = models["volregime_rf"].predict(X_sample)[0]
    vol_proba = models["volregime_rf"].predict_proba(X_sample)[0, int(vol_pred)]
    vol_label = "🌪 High volatility" if vol_pred == 1 else "🌤 Normal volatility"

    next_ret_pred = models["nextret_rf"].predict(X_sample)[0]
    
    # --- KPI metrics ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
        "Directional Bias (Probability)",
        f"{dir_proba_up:.1%}",
        "Bullish" if dir_proba_up >= 0.55 else ("Bearish/Neutral" if dir_proba_up <= 0.45 else "Mixed"),
    )
    with col2:
        st.metric(
        "Volatility Context",
        vol_label,
        f"Model confidence {vol_proba:.1%}",
    )
    with col3:
        st.metric(
        "Expected Next-Day Move (Return)",
        f"{next_ret_pred:.3f}",
        "Higher = bigger expected move",
    )
        
    st.markdown("### What these numbers usually mean")

    a, b, c = st.columns(3)
    with a:
        st.caption("**Directional Bias**")
        st.write(
        "- ~50% = no edge\n"
        "- 55%+ = mild bullish tilt\n"
        "- 60%+ = strong signal (still not a guarantee)"
        )
    with b:
        st.caption("**Volatility Context**")
        st.write(
        "- Normal = calmer tape\n"
        "- High vol = wider swings, harder risk control\n"
        "- High vol + weak bias = be cautious"
        )
    with c:
        st.caption("**Expected Move**")
        st.write(
        "- Near 0 = noise / tiny move\n"
        "- Bigger magnitude = bigger expected swing\n"
        "- Use for ranking, not precision"
    )

    # Hide the huge engineered feature row by default
    with st.expander("Selected Sweep (engineered features)", expanded=False):
        st.dataframe(sample[feature_cols], use_container_width=True, height=220)

    # ---------- Manual Sweep Input ----------
    st.subheader("What-If Scenario Analysis")

    with st.expander("What this panel does", expanded=False):
        st.markdown(
            """
- Starts from the selected sweep row (engineered features)
- Lets you tweak a few knobs (Spot/Strike/Spread/Flow)
- Re-runs the same trained models
- Returns a plain-English interpretation (if OpenAI key is set)
"""
        )

    base_row = sample[feature_cols].copy()

    editable_features = ["Spot", "Strike", "spread_pct", "flow_intensity"]
    manual_inputs = {}

    cols = st.columns(len(editable_features))
    for col, feat in zip(cols, editable_features):
        with col:
            if feat not in base_row.columns:
                st.write(f"⚠️ Missing feature: `{feat}`")
                manual_inputs[feat] = None
                continue

            original_val = float(base_row.iloc[0][feat])
            manual_val = st.number_input(
                feat,
                value=original_val,
                step=abs(original_val) * 0.01 if original_val != 0 else 0.01,
                format="%.6f",
            )
            manual_inputs[feat] = manual_val
            base_row.iloc[0][feat] = manual_val

    run_manual = st.button("Run Manual Prediction", type="primary")

    st.markdown("---")
    st.markdown("### Manual Input Predictions")

    if run_manual:
        X_manual = base_row[feature_cols]

        manual_dir_proba_up = models["direction_rf"].predict_proba(X_manual)[0, 1]
        manual_dir_label = "⬆️ Up" if manual_dir_proba_up >= 0.5 else "⬇️ Down / Flat"

        manual_vol_pred = models["volregime_rf"].predict(X_manual)[0]
        manual_vol_proba = models["volregime_rf"].predict_proba(X_manual)[0, int(manual_vol_pred)]
        manual_vol_label = "🌪 High volatility" if manual_vol_pred == 1 else "🌤 Normal volatility"

        manual_next_ret_pred = float(models["nextret_rf"].predict(X_manual)[0])

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
        "Directional Bias (Probability) — What-If",
        f"{manual_dir_proba_up:.1%}",
        "Bullish" if manual_dir_proba_up >= 0.55 else ("Bearish/Neutral" if manual_dir_proba_up <= 0.45 else "Mixed"),
    )
        with c2:
            st.metric(
        "Volatility Context — What-If",
        manual_vol_label,
        f"Model confidence {manual_vol_proba:.1%}",
    )
        with c3:
            st.metric(
        "Expected Next-Day Move — What-If",
        f"{manual_next_ret_pred:.3f}",
        "Higher = bigger expected move",
    )

        st.markdown("#### AI interpretation of this manual sweep")
        ai_text = generate_manual_ai_summary(
            manual_inputs=manual_inputs,
            dir_proba_up=manual_dir_proba_up,
            vol_label=manual_vol_label,
            vol_proba=manual_vol_proba,
            next_ret_pred=manual_next_ret_pred,
        )
        st.write(ai_text)
    else:
        st.info("Adjust the fields above and click **Run Manual Prediction**.")

    # ---------- What drove this prediction ----------
    st.subheader("What Drove This Prediction")

    top_dir = get_top_features(models["direction_rf"], feature_cols, k=5)
    top_vol = get_top_features(models["volregime_rf"], feature_cols, k=5)
    top_ret = get_top_features(models["nextret_rf"], feature_cols, k=5)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Direction – key signals**")
        if top_dir:
            st.bar_chart(pd.DataFrame(top_dir, columns=["feature", "importance"]).set_index("feature"))
        else:
            st.caption("No feature importances available.")
    with c2:
        st.markdown("**Vol regime – key signals**")
        if top_vol:
            st.bar_chart(pd.DataFrame(top_vol, columns=["feature", "importance"]).set_index("feature"))
        else:
            st.caption("No feature importances available.")
    with c3:
        st.markdown("**Return – key signals**")
        if top_ret:
            st.bar_chart(pd.DataFrame(top_ret, columns=["feature", "importance"]).set_index("feature"))
        else:
            st.caption("No feature importances available.")


# ============================
# DIAGNOSTICS VIEW
# ============================
else:
    st.subheader("🔧 Model Diagnostics (Advanced)")
    st.caption("Nothing heavy runs unless you explicitly turn it on.")

    with st.expander("Paths (sanity checks)", expanded=False):
        st.code(
            f"PROJECT_ROOT: {PROJECT_ROOT}\n"
            f"MODELS_DIR:   {MODELS_DIR}\n"
            f"DATA_PATH:    {DATA_PATH}",
            language="bash",
        )

    run_diag = st.checkbox("Run evaluation metrics (may take a few seconds)", value=False)

    if not run_diag:
        st.info("Enable the checkbox above to compute diagnostics.")
    else:
        df = load_training_data()
        feature_cols = get_feature_cols(df)

        eval_df = df.sample(min(1000, len(df)), random_state=0)
        X_eval = eval_df[feature_cols]

        tab_dir, tab_vol, tab_ret = st.tabs(
            ["Direction model", "Volatility regime model", "Return regression"]
        )

        with tab_dir:
            st.markdown("#### Direction model (next move up vs. down)")
            y_true_dir = eval_df["direction_up"]
            y_pred_dir = models["direction_rf"].predict(X_eval)

            cm_dir = confusion_matrix(y_true_dir, y_pred_dir, labels=[0, 1])
            acc_dir = (cm_dir.trace() / cm_dir.sum()) if cm_dir.sum() > 0 else 0.0

            st.write(f"**Accuracy:** `{acc_dir:.1%}`")
            st.dataframe(
                pd.DataFrame(
                    cm_dir,
                    index=["Actual 0 (down/flat)", "Actual 1 (up)"],
                    columns=["Pred 0 (down/flat)", "Pred 1 (up)"],
                )
            )

            dir_importances = pd.DataFrame(
                {"feature": feature_cols, "importance": models["direction_rf"].feature_importances_}
            ).sort_values("importance", ascending=False)
            st.write("Top features (direction):")
            st.bar_chart(dir_importances.head(12).set_index("feature"))

        with tab_vol:
            st.markdown("#### Volatility regime model (normal vs high-vol)")
            y_true_vol = eval_df["vol_regime"]
            y_pred_vol = models["volregime_rf"].predict(X_eval)

            cm_vol = confusion_matrix(y_true_vol, y_pred_vol, labels=[0, 1])
            acc_vol = (cm_vol.trace() / cm_vol.sum()) if cm_vol.sum() > 0 else 0.0

            st.write(f"**Accuracy:** `{acc_vol:.1%}`")
            st.dataframe(
                pd.DataFrame(
                    cm_vol,
                    index=["Actual 0 (normal)", "Actual 1 (high-vol)"],
                    columns=["Pred 0 (normal)", "Pred 1 (high-vol)"],
                )
            )

            vol_importances = pd.DataFrame(
                {"feature": feature_cols, "importance": models["volregime_rf"].feature_importances_}
            ).sort_values("importance", ascending=False)
            st.write("Top features (vol regime):")
            st.bar_chart(vol_importances.head(12).set_index("feature"))

        with tab_ret:
            st.markdown("#### Return regression (next_return_1d)")
            y_true_ret = eval_df["next_return_1d"]
            y_pred_ret = models["nextret_rf"].predict(X_eval)

            mae = mean_absolute_error(y_true_ret, y_pred_ret)
            rmse = np.sqrt(mean_squared_error(y_true_ret, y_pred_ret))

            c_mae, c_rmse = st.columns(2)
            with c_mae:
                st.metric("MAE", f"{mae:.4f}")
            with c_rmse:
                st.metric("RMSE", f"{rmse:.4f}")

            ret_importances = pd.DataFrame(
                {"feature": feature_cols, "importance": models["nextret_rf"].feature_importances_}
            ).sort_values("importance", ascending=False)
            st.write("Top features (return):")
            st.bar_chart(ret_importances.head(12).set_index("feature"))
