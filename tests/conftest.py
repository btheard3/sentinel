# tests/conftest.py
from pathlib import Path

import pandas as pd
import pytest

from src.config import (
    PROCESSED_TRADYFLOW_PATH,
    JOURNAL_DIR,  # keeps imports future-proof
)

@pytest.fixture(scope="session")
def modeling_df() -> pd.DataFrame:
    """Full modeling dataset from Notebook 02."""
    path = Path("data/processed/tradyflow_modeling.parquet")
    if not path.exists():
        path = Path(PROCESSED_TRADYFLOW_PATH)  # fallback
    df = pd.read_parquet(path)
    # sanity: this is your “feature engineering complete” dataset
    assert not df.empty
    return df


@pytest.fixture(scope="session")
def training_df() -> pd.DataFrame:
    """Training dataset with targets from Notebook 03."""
    path = Path("data/processed/tradyflow_training.parquet")
    df = pd.read_parquet(path)
    assert not df.empty
    return df