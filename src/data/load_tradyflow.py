# src/data/load_tradyflow.py

from pathlib import Path
from typing import Union

import pandas as pd

from src.config import RAW_TRADYFLOW_PATH, PROCESSED_TRADYFLOW_PATH


PathLike = Union[str, Path]


def load_tradyflow_raw(path: PathLike = RAW_TRADYFLOW_PATH) -> pd.DataFrame:
    """
    Load the raw TradyFlow CSV dataset.

    Parameters
    ----------
    path : str or Path
        Location of the raw CSV file. Defaults to config.RAW_TRADYFLOW_PATH.

    Returns
    -------
    pd.DataFrame
        Raw TradyFlow trades, exactly as provided by the source.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw TradyFlow file not found at: {path}")

    return pd.read_csv(path)


def load_tradyflow_processed(path: PathLike = PROCESSED_TRADYFLOW_PATH) -> pd.DataFrame:
    """
    Load the processed TradyFlow dataset used for feature engineering and modeling.

    Parameters
    ----------
    path : str or Path
        Location of the processed Parquet file. Defaults to config.PROCESSED_TRADYFLOW_PATH.

    Returns
    -------
    pd.DataFrame
        Cleaned TradyFlow trades ready for feature engineering.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed TradyFlow file not found at: {path}")

    return pd.read_parquet(path)