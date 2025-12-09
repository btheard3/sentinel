# tests/test_app_wiring.py

from pathlib import Path
import pandas as pd
import joblib

from sentinel_app.app import load_models, load_training_data, get_feature_cols, get_top_features, MODELS_DIR, DATA_PATH


def test_models_load_and_methods():
    """Ensure models load and expose .predict and .predict_proba correctly."""
    models = load_models()

    assert "direction_rf" in models
    assert "volregime_rf" in models
    assert "nextret_rf" in models

    # Check interfaces
    assert hasattr(models["direction_rf"], "predict_proba")
    assert hasattr(models["volregime_rf"], "predict_proba")
    assert hasattr(models["nextret_rf"], "predict")

    # Confirm no broken files
    for name, model in models.items():
        assert model is not None

def test_feature_cols_match_training_logic():
    df = load_training_data()
    feature_cols = get_feature_cols(df)

    # They must be numeric
    assert all(df[c].dtype in ("float64", "int64") for c in feature_cols)

    # They must NOT contain labels
    forbidden = {"direction_up", "vol_regime", "next_return_1d", "next_spot"}
    assert forbidden.isdisjoint(feature_cols)

    # Must return >5 features
    assert len(feature_cols) > 5 

def test_sample_inference():
    df = load_training_data()
    feature_cols = get_feature_cols(df)
    models = load_models()

    sample = df[feature_cols].sample(5, random_state=42)

    # Direction
    dir_proba = models["direction_rf"].predict_proba(sample)
    assert dir_proba.shape == (5, 2)

    # Vol Regime
    vol_preds = models["volregime_rf"].predict(sample)
    assert set(vol_preds).issubset({0, 1})

    # Return Regression
    ret_pred = models["nextret_rf"].predict(sample)
    assert len(ret_pred) == 5

import pytest

def test_missing_feature_throws_error():
    df = load_training_data()
    feature_cols = get_feature_cols(df)
    models = load_models()

    broken = df[feature_cols].sample(1).copy()
    broken = broken.drop(columns=feature_cols[0])   # delete one feature

    with pytest.raises(Exception):
        models["direction_rf"].predict_proba(broken)

def test_model_feature_importance_length_matches_feature_cols():
    df = load_training_data()
    feature_cols = get_feature_cols(df)
    models = load_models()

    assert len(models["direction_rf"].feature_importances_) == len(feature_cols)
    assert len(models["volregime_rf"].feature_importances_) == len(feature_cols)
    assert len(models["nextret_rf"].feature_importances_) == len(feature_cols)

def test_get_top_features_returns_sorted_pairs():
    df = load_training_data()
    feature_cols = get_feature_cols(df)
    models = load_models()

    top_feats = get_top_features(models["direction_rf"], feature_cols, k=5)

    # Should return k (or fewer if k>len) (name, importance) pairs
    assert 1 <= len(top_feats) <= 5
    names, importances = zip(*top_feats)

    # Names come from the feature columns
    assert set(names).issubset(set(feature_cols))

    # Importances are non-negative and sorted descending
    assert all(i >= 0 for i in importances)
    assert list(importances) == sorted(importances, reverse=True)