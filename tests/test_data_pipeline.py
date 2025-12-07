# tests/test_data_pipeline.py
import numpy as np
from pandas import DataFrame

from src.features.feature_builder import build_basic_features



def test_feature_builder_shapes(modeling_df: DataFrame):
    df_feat = build_basic_features(modeling_df)

    # 1) Should not shrink the dataset
    assert len(df_feat) == len(modeling_df)

    # 2) New features should exist
    expected_cols = {
        "moneyness",
        "spread_pct",
        "flow_intensity",
        "log_vol",
        "log_prems",
        "dte",
        "is_call",
        "is_put",
    }

    missing = expected_cols.difference(df_feat.columns)
    assert not missing, f"Missing engineered columns: {missing}"

    # 3) No NaNs in engineered features
    engineered_cols = list(expected_cols)  # sets can't be used as indexers
    assert df_feat[engineered_cols].isna().sum().sum() == 0



def test_feature_value_ranges(modeling_df: DataFrame):
    df_feat = build_basic_features(modeling_df)

    # Most flows should be near ATM
    near_atm_share = df_feat["moneyness"].between(-0.5, 0.5).mean()
    assert near_atm_share > 0.90

    # DTE sanity: allow a few bad rows, but not many
    neg_share = (df_feat["dte"] < 0).mean()
    assert neg_share < 0.05   # < 5% of rows with negative DTE

    # 99% of trades within ~4 years – catches broken date math, allows LEAPs
    assert df_feat["dte"].quantile(0.99) < 4 * 365
