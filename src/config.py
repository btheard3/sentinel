import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Basic config for Sentinel
PROJECT_NAME = "Sentinel – Premarket Forecasting Engine"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
JOURNAL_DIR = os.path.join(BASE_DIR, "data", "journal")

# API keys
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

if not POLYGON_API_KEY:
    print("[config] Warning: POLYGON_API_KEY is not set – live calls will fail.")

# Project root (sentinel/)
BASE_DIR = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
JOURNAL_DIR = DATA_DIR / "journal"

# TradyFlow file paths
RAW_TRADYFLOW_PATH = RAW_DIR / "tradyflow_options.csv"
PROCESSED_TRADYFLOW_PATH = PROCESSED_DIR / "tradyflow_clean.parquet"
