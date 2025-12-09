# tests/test_data_guards.py

import numpy as np
from sentinel_app.app import load_training_data, get_feature_cols


def test_training_schema_and_size():
    """Guardrail: dataset should look like our current modeling snapshot."""
    df = load_training_data()

    # Expected columns that must always exist
    required_cols = {
        "Spot",
        "BidAsk",
        "Orders",
        "Vol",
        "Prems",
        "Diff(%)",
        "ITM",
        "moneyness",
        "spread_pct",
        "flow_intensity",
        "log_vol",
        "log_prems",
        "dte",
        "direction_up",
        "vol_regime",
        "next_return_1d",
    }

    assert required_cols.issubset(df.columns)

    # Row count should stay in a reasonable band
    n_rows = len(df)
    assert 5_000 <= n_rows <= 10_000, f"Unexpected row count: {n_rows}"

    # Feature columns must be non-empty
    feature_cols = get_feature_cols(df)
    assert len(feature_cols) >= 10


def test_label_balance_direction_up():
    """Guardrail: direction_up should not collapse to one class."""
    df = load_training_data()

    share_up = df["direction_up"].mean()  # since it's 0/1
    # From Notebook 03 we saw ~0.47–0.53, give it a healthy band:
    assert 0.35 < share_up < 0.65, f"direction_up share_out_of_band={share_up:.3f}"


def test_label_balance_vol_regime():
    """Guardrail: vol_regime should have a sane fraction of high-vol rows."""
    df = load_training_data()

    share_high = df["vol_regime"].mean()
    # We expect ~25% high-vol; keep a wide but protective band:
    assert 0.10 < share_high < 0.40, f"vol_regime share_out_of_band={share_high:.3f}"


def test_key_feature_sanity_ranges():
    """Guardrail: core engineered features stay in plausible ranges."""
    df = load_training_data()

    # Moneyness: slightly OTM on average, not wild
    mn = df["moneyness"].mean()
    assert -0.5 < mn < 0.5

    # Spread percentage: mostly tight < 0.6
    assert df["spread_pct"].between(0, 0.6).mean() > 0.95

    # DTE sanity:
    # - Allow zero (same-day expiry)
    # - Almost all should be >= 0
    # - Median should be reasonably small (we're not trading 10-year options)
    assert (df["dte"] >= 0).mean() > 0.99   # at most 1% weird negatives
    assert df["dte"].median() < 365 * 2     # typically within 2 years
