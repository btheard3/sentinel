"""
Utilities to log Sentinel signals to an Excel trade journal.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from ..config import JOURNAL_DIR

DEFAULT_JOURNAL = Path(JOURNAL_DIR) / "sentinel_trade_journal.xlsx"

COLUMNS = [
    "date",
    "ticker",
    "direction",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "size",
    "model_prob_continuation",
    "expected_return",
    "actual_return",
    "setup_tag",
    "notes_llm",
]

def append_trades(df_trades: pd.DataFrame, path: Optional[Path] = None) -> Path:
    """
    Append trades to the Excel journal. Creates the file if missing.
    """
    journal_path = path or DEFAULT_JOURNAL
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    if journal_path.exists():
        existing = pd.read_excel(journal_path)
        combined = pd.concat([existing, df_trades[COLUMNS]], ignore_index=True)
    else:
        combined = df_trades[COLUMNS]

    combined.to_excel(journal_path, index=False)
    return journal_path
