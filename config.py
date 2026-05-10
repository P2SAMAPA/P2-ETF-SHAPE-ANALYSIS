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

# --- Macro features (compatibility) ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- Shape detection parameters ---
LOOKBACK_YEARS = 20          # Use full data from 2008 onwards
TROUGH_WINDOW = 5            # days to define local min
PEAK_THRESHOLD = 0.10        # recovery defined as +10% from trough
MIN_RECOVERY_DAYS = 5        # minimum segment length
INTERP_POINTS = 50           # number of points after interpolation
N_CLUSTERS = 3               # V, U, L (configurable)
CLUSTER_NAMES = ["V", "U", "L"]

PROCRUSTES_ITER = 10         # alignment iterations

# --- Output ---
TODAY = datetime.now().strftime("%Y-%m-%d")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
