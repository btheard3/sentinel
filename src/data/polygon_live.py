"""
Functions for pulling live / premarket data from Polygon.
Tonight this is just a skeleton – we will flesh it out later.
"""

from typing import Optional
import pandas as pd
import requests
from ..config import POLYGON_API_KEY

BASE_URL = "https://api.polygon.io"

def _get(session: Optional[requests.Session], path: str, params: dict) -> dict:
    if not POLYGON_API_KEY:
        raise RuntimeError("POLYGON_API_KEY not set.")
    url = f"{BASE_URL}{path}"
    params = {**params, "apiKey": POLYGON_API_KEY}
    s = session or requests.Session()
    resp = s.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

def get_placeholder_premarket_df() -> pd.DataFrame:
    """
    Temporary stub used while we design the pipeline.
    Returns a tiny DataFrame with fake premarket movers.
    """
    data = [
        {"ticker": "AMD", "premarket_gap_pct": 3.2, "premarket_volume": 1500000},
        {"ticker": "NVDA", "premarket_gap_pct": -1.1, "premarket_volume": 900000},
    ]
    return pd.DataFrame(data)
