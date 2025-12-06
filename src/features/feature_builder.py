# src/features/feature_builder.py

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def _parse_number_with_suffix(series: pd.Series) -> pd.Series:
    """
    Convert strings like '183.60K', '4.07K', '500', '1.01M' to numeric floats.

    Rules:
      - Remove commas and spaces
      - Extract numeric part and optional suffix (K/M)
      - K => * 1e3, M => * 1e6
    """
    s = series.astype(str).str.replace(",", "", regex=False).str.strip()

    # Extract suffix and numeric part
    suffix = s.str.extract(r"([KM])$", expand=False)
    base = s.str.extract(r"([\d\.]+)", expand=False)

    num = pd.to_numeric(base, errors="coerce")
    factor = suffix.map({"K": 1e3, "M": 1e6}).fillna(1.0)

    return num * factor


def build_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build v1 feature set for Sentinel from the TradyFlow dataset.

    Input columns (from EDA):
      Time, Sym, C/P, Exp, Strike, Spot, BidAsk,
      Orders, Vol, Prems, OI, Diff(%), ITM

    Returns a DataFrame with:
      - original columns + parsed time/expiration
      - engineered numeric features suitable for modeling
    """
    # Work on a copy
    df_feat = df.copy()

    # ---- 1) Ensure datetime columns are parsed ----
    # We ALWAYS parse here so we don't depend on upstream types.
    df_feat["Time_dt"] = pd.to_datetime(df_feat["Time"], errors="coerce")
    df_feat["Exp_dt"] = pd.to_datetime(df_feat["Exp"], errors="coerce")

    # ---- 2) Convert string-based numeric columns to floats ----
    for col in ["Vol", "Prems", "OI"]:
        df_feat[col] = _parse_number_with_suffix(df_feat[col]).fillna(0.0)

    # ---- 3) Core engineered features ----

    # Moneyness: how far in/out of the money the contract is
    df_feat["moneyness"] = (df_feat["Spot"] - df_feat["Strike"]) / df_feat["Spot"]

    # Bid-ask spread as % of spot (liquidity proxy)
    df_feat["spread_pct"] = df_feat["BidAsk"] / df_feat["Spot"]

    # Flow intensity: premium-weighted volume
    df_feat["flow_intensity"] = df_feat["Vol"] * df_feat["Prems"]

    # Log-transformed volume and premium to reduce skew
    df_feat["log_vol"] = np.log1p(df_feat["Vol"])
    df_feat["log_prems"] = np.log1p(df_feat["Prems"])

    # Days to expiration (DTE) – now based on the parsed datetime columns
    df_feat["dte"] = (df_feat["Exp_dt"] - df_feat["Time_dt"]).dt.days

    # Encode call/put and ITM as numeric flags
    cp_upper = df_feat["C/P"].astype(str).str.upper()
    df_feat["is_call"] = (cp_upper == "CALL").astype(int)
    df_feat["is_put"] = (cp_upper == "PUT").astype(int)
    df_feat["ITM"] = df_feat["ITM"].astype(int)

    return df_feat
