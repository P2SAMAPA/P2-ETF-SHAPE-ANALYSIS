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
    """Return wide‑format closing prices for given tickers."""
    if not tickers:
        return pd.DataFrame()
    available = [t for t in tickers if t in df_wide.columns]
    if not available:
        raise ValueError(f"None of the tickers {tickers} found in data.")
    # Ensure Date column exists and set as index
    if 'Date' not in df_wide.columns:
        df_wide = df_wide.reset_index()
    return df_wide[['Date'] + available].set_index('Date')[available].dropna(how='all')

def prepare_macro_features(df_wide: pd.DataFrame) -> pd.DataFrame:
    macro_cols = [c for c in config.MACRO_COLS if c in df_wide.columns]
    macro_df = df_wide[['Date'] + macro_cols].copy()
    macro_df = macro_df.set_index('Date').ffill().dropna()
    return macro_df
