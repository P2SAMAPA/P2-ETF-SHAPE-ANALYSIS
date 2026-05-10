# P2-ETF-SHAPE-ANALYSIS

**Morphological shape analysis** of ETF recovery patterns using Kendall's shape space and Procrustes distance. Classifies shapes as **V, U, or L** and provides trading signals.

## Features

- Detects recovery segments (trough → subsequent peak).
- Normalises time and price, interpolates to fixed length.
- Aligns shapes via Procrustes superimposition.
- Clusters into 3 shape types (configurable).
- Outputs current shape, confidence, and distribution.
- Dashboard shows aligned shape plots.

## Data

Uses `P2SAMAPA/fi-etf-macro-signal-master-data` (2008–present). Results pushed to `P2SAMAPA/p2-etf-shape-analysis-results`.

## Installation

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-SHAPE-ANALYSIS.git
cd P2-ETF-SHAPE-ANALYSIS
pip install -r requirements.txt
