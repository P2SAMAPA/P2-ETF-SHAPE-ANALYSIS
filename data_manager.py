import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data() -> pd.DataFrame:
    print(f"Downloading {config.DATA_FILE} from {config.DATA_REPO}...")
    file_path = hf_hub_download(
        repo_id=config.DATA_REPO,
        filename=config.DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
        cache_dir="./hf_cache"
    )
    df = pd.read_parquet(file_path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_prices_matrix(df_wide: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Prepare wide‑format closing prices for given tickers."""
    # First ensure we have 'Ticker' column? The input df_wide may have columns 'Date', 'Ticker', 'Close'
    if 'Ticker' in df_wide.columns and 'Close' in df_wide.columns:
        # Long format, pivot
        pivot = df_wide.pivot(index='Date', columns='Ticker', values='Close')
    else:
        # Already wide? Assume index is date and columns are tickers
        pivot = df_wide
    # Keep only requested tickers that exist
    available = [t for t in tickers if t in pivot.columns]
    if not available:
        return pd.DataFrame()
    return pivot[available].dropna(how='all')

def prepare_macro_features(df_wide: pd.DataFrame) -> pd.DataFrame:
    macro_cols = [c for c in config.MACRO_COLS if c in df_wide.columns]
    macro_df = df_wide[['Date'] + macro_cols].copy()
    macro_df = macro_df.set_index('Date').ffill().dropna()
    return macro_df
