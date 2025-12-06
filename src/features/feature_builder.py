# src/features/feature_builder.py

from __future__ import annotations

from typing import Literal

import pandas as pd


def _parse_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Time and Exp columns from strings to pandas datetime.
    Assumes:
      - Time: includes date + time (e.g. '6/17/2022 15:07')
      - Exp: date string (e.g. '10/21/2022')
    """
    df = df.copy()

    if df["Time"].dtype == "object":
        df["Time"] = pd.to_datetime(df["Time"])
    if df["Exp"].dtype == "object":
        df["Exp"] = pd.to_datetime(df["Exp"])

    return df


def build_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build v1 feature set for Sentinel from the TradyFlow dataset.

    Input columns (from EDA):
      Time, Sym, C/P, Exp, Strike, Spot, BidAsk,
      Orders, Vol, Prems, OI, Diff(%), ITM

    Returns a DataFrame with:
      - original columns (with parsed datetimes)
      - engineered numeric features suitable for modeling
    """
    df = _parse_datetime_columns(df)

    # Copy to avoid modifying caller's frame
    df_feat = df.copy()

    # 1) Moneyness: how far in/out of the money the contract is
    df_feat["moneyness"] = (df_feat["Spot"] - df_feat["Strike"]) / df_feat["Spot"]

    # 2) Bid-ask spread as % of spot (liquidity proxy)
    df_feat["spread_pct"] = df_feat["BidAsk"] / df_feat["Spot"]

    # 3) Flow intensity: premium-weighted volume
    #    (rough proxy for "how much money is flowing into this contract")
    df_feat["flow_intensity"] = df_feat["Vol"] * df_feat["Prems"]

    # 4) Log-transformed volume and premium to reduce skew
    df_feat["log_vol"] = (df_feat["Vol"].replace(0, 1)).pipe(lambda s: (s).map(float)).rpow(1)  # keep positive
    df_feat["log_vol"] = (df_feat["Vol"].replace(0, 1)).apply(lambda x: float(pd.np.log1p(x)))  # type: ignore

    df_feat["log_prems"] = (df_feat["Prems"].replace(0, 1)).apply(lambda x: float(pd.np.log1p(x)))  # type: ignore

    # 5) Days to expiration (DTE)
    df_feat["dte"] = (df_feat["Exp"].dt.date - df_feat["Time"].dt.date).dt.days

    # 6) Encode call/put and ITM as numeric flags
    df_feat["is_call"] = (df_feat["C/P"].str.upper() == "Call").astype(int)
    df_feat["is_put"] = (df_feat["C/P"].str.upper() == "Put").astype(int)

    # ITM already 0/1 int64, just ensure type
    df_feat["ITM"] = df_feat["ITM"].astype(int)

    return df_feat