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
    all_results = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} ===")
        # Get prices for these tickers only
        uni_prices = data_manager.prepare_prices_matrix(df, tickers)
        if uni_prices.empty:
            print(f"  No price data for universe, skipping.")
            continue

        universe_results = {}
        for ticker in tickers:
            if ticker not in uni_prices.columns:
                continue
            price_series = uni_prices[ticker].dropna()
            if len(price_series) < 100:
                print(f"  {ticker}: insufficient data ({len(price_series)} days)")
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
                print(f"  {ticker}: {result['error']}")
                continue

            all_segments = result["segments"]
            if len(all_segments) == 0:
                print(f"  {ticker}: no recovery segments found")
                continue

            last_segment = all_segments[-1]
            closest_cluster, dist = sa.classify_shape(last_segment, centers)
            shape_name = result["cluster_names"].get(closest_cluster, "unknown")
            confidence = 1.0 / (1.0 + dist) if dist < 1e8 else 0.0

            universe_results[ticker] = {
                "current_shape": shape_name,
                "confidence": float(confidence),
                "procrustes_distance": float(dist),
                "num_recoveries": len(all_segments),
                "cluster_distribution": {str(k): int((labels == k).sum()) for k in range(config.N_CLUSTERS)},
                "cluster_names": result["cluster_names"],
                "last_recovery_normalized": last_segment.tolist()
            }
        if universe_results:
            all_results[universe_name] = universe_results

    if not all_results:
        print("No results generated for any universe.")
        return

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/shape_analysis_{config.TODAY}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": config.TODAY, "universes": all_results}, f, indent=2)

    push_results.push_daily_result(local_path)
    print("\n=== Shape analysis complete ===")

if __name__ == "__main__":
    main()
