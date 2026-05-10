"""
Configuration for P2-ETF-SHAPE-ANALYSIS engine.
"""

import os
from datetime import datetime

# --- Hugging Face ---
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
DATA_FILE = "master_data.parquet"
OUTPUT_REPO = "P2SAMAPA/p2-etf-shape-analysis-results"

# --- Universe definitions ---
FI_COMMODITIES = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
    "GDX", "XME", "IWF", "XSD", "XBI", "IWM"
]
COMBINED = list(set(FI_COMMODITIES + EQUITY_SECTORS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES,
    "EQUITY_SECTORS": EQUITY_SECTORS,
    "COMBINED": COMBINED
}

MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- Shape detection parameters ---
TROUGH_WINDOW = 5
PEAK_THRESHOLD = 0.08          # 8% – captures stronger recoveries
MIN_RECOVERY_DAYS = 5
INTERP_POINTS = 50
N_CLUSTERS = 3                 # V, U, L
PROCRUSTES_ITER = 10

# --- Output ---
TODAY = datetime.now().strftime("%Y-%m-%d")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
