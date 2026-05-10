"""
Main training script for shape analysis.
For each ETF, detect recovery segments, cluster shapes, and classify the most recent incomplete recovery.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import config
import data_manager
from shape_analyzer import ShapeAnalyzer
import push_results

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    prices = data_manager.prepare_prices_matrix(df, [])  # will be updated later per universe
    all_results = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} ===")
        # Filter prices for the current universe
        uni_prices = prices[tickers] if all(t in prices.columns for t in tickers) else pd.DataFrame()
        if uni_prices.empty:
            # Fallback: prepare using data_manager with specific tickers
            uni_prices = data_manager.prepare_prices_matrix(df, tickers)
        if uni_prices.empty:
            continue

        universe_results = {}
        for ticker in tickers:
            if ticker not in uni_prices.columns:
                continue
            price_series = uni_prices[ticker].dropna()
            if len(price_series) < 100:
                print(f"  {ticker}: insufficient data ({len(price_series)} points)")
                continue

            sa = ShapeAnalyzer(
                trough_window=config.TROUGH_WINDOW,
                peak_threshold=config.PEAK_THRESHOLD,
                min_recovery_days=config.MIN_RECOVERY_DAYS,
                interp_points=config.INTERP_POINTS,
                n_clusters=config.N_CLUSTERS,
                procrustes_iter=config.PROCRUSTES_ITER
            )
            result, centers, labels = sa.analyze(price_series)
            if "error" in result:
                print(f"  {ticker}: {result['error']} (recoveries={result.get('num_recoveries',0)})")
                continue
            else:
                print(f"  {ticker}: {result['num_recoveries']} recoveries, {len(labels)} clusters")

            # Classify the most recent recovery (last segment)
            all_segments = result["segments"]
            if len(all_segments) == 0:
                continue
            last_segment = all_segments[-1]
            closest_cluster, dist = sa.classify_shape(last_segment, centers)
            shape_name = result["cluster_names"].get(closest_cluster, "unknown")
            confidence = 1.0 / (1.0 + dist) if dist < 1e8 else 0.0

            universe_results[ticker] = {
                "current_shape": shape_name,
                "confidence": float(confidence),
                "procrustes_distance": float(dist),
                "num_recoveries": result["num_recoveries"],
                "cluster_distribution": {str(k): int((labels == k).sum()) for k in range(config.N_CLUSTERS)},
                "cluster_names": result["cluster_names"],
                "last_recovery_normalized": last_segment.tolist()
            }
        all_results[universe_name] = universe_results

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/shape_analysis_{config.TODAY}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": config.TODAY, "universes": all_results}, f, indent=2)

    push_results.push_daily_result(local_path)
    print("\n=== Shape analysis complete ===")

if __name__ == "__main__":
    main()
