"""
Shape analysis using Kendall's shape space and Procrustes distance.
Detects recovery segments, aligns them, classifies into V/U/L shapes.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, squareform, cdist
from collections import Counter
import warnings

class ShapeAnalyzer:
    def __init__(self, trough_window=5, peak_threshold=0.05, min_recovery_days=5,
                 interp_points=50, n_clusters=3, procrustes_iter=10):
        self.trough_window = trough_window
        self.peak_threshold = peak_threshold
        self.min_recovery_days = min_recovery_days
        self.interp_points = interp_points
        self.n_clusters = n_clusters
        self.procrustes_iter = procrustes_iter
        self.cluster_centers_ = None
        self.cluster_labels_ = None

    def find_recovery_segments(self, prices):
        """prices: pd.Series indexed by datetime."""
        segments = []
        values = prices.values
        n = len(values)

        # Find troughs (local minima over window)
        trough_idx = []
        for i in range(self.trough_window, n - self.trough_window):
            window = values[i - self.trough_window : i + self.trough_window + 1]
            if values[i] == min(window):
                trough_idx.append(i)

        if not trough_idx:
            return []

        for ti in trough_idx:
            trough_price = values[ti]
            max_lookahead = min(n, ti + 252)  # up to one year
            peak_candidate = None
            for j in range(ti+1, max_lookahead):
                if values[j] >= trough_price * (1 + self.peak_threshold):
                    peak_candidate = j
                    break
            if peak_candidate is None:
                continue
            segment_prices = values[ti:peak_candidate+1]
            if len(segment_prices) < self.min_recovery_days:
                continue
            # Normalise time to [0,1]
            time_norm = np.linspace(0, 1, len(segment_prices))
            # Normalise price to [0,1]
            pmin = segment_prices.min()
            pmax = segment_prices.max()
            if pmax == pmin:
                continue
            price_norm = (segment_prices - pmin) / (pmax - pmin)
            # Interpolate to fixed number of points
            interp_time = np.linspace(0, 1, self.interp_points)
            f = interp1d(time_norm, price_norm, kind='linear', fill_value='extrapolate')
            interp_norm = f(interp_time)
            interp_norm = np.clip(interp_norm, 0, 1)
            segment = np.column_stack([interp_time, interp_norm])
            segments.append(segment)
        return segments

    def _procrustes_align(self, X, Y):
        """Procrustes superimposition: translate, scale, rotate Y to best fit X."""
        X_cent = X - X.mean(axis=0)
        Y_cent = Y - Y.mean(axis=0)
        size_X = np.sqrt(np.sum(X_cent**2))
        size_Y = np.sqrt(np.sum(Y_cent**2))
        if size_X < 1e-8 or size_Y < 1e-8:
            return Y, np.inf
        X_scaled = X_cent / size_X
        Y_scaled = Y_cent / size_Y
        M = Y_scaled.T @ X_scaled
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        Y_rotated = Y_scaled @ R
        dist = np.sum((X_scaled - Y_rotated)**2)
        return Y_rotated, dist

    def compute_procrustes_matrix(self, segments):
        n = len(segments)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                _, dist = self._procrustes_align(segments[i], segments[j])
                D[i,j] = D[j,i] = dist
        return D

    def _kmedoids(self, D, k, max_iter=100):
        n = D.shape[0]
        np.random.seed(42)
        medoids = np.random.choice(n, k, replace=False)
        labels = np.argmin(D[:, medoids], axis=1)
        for _ in range(max_iter):
            new_medoids = medoids.copy()
            for c in range(k):
                cluster_idx = np.where(labels == c)[0]
                if len(cluster_idx) == 0:
                    continue
                sum_dist = D[cluster_idx][:, cluster_idx].sum(axis=0)
                best = cluster_idx[np.argmin(sum_dist)]
                new_medoids[c] = best
            new_labels = np.argmin(D[:, new_medoids], axis=1)
            if np.array_equal(new_labels, labels) and np.array_equal(new_medoids, medoids):
                break
            medoids = new_medoids
            labels = new_labels
        return labels, medoids

    def cluster_shapes(self, segments):
        if len(segments) < self.n_clusters:
            return None, None
        D = self.compute_procrustes_matrix(segments)
        labels, medoid_indices = self._kmedoids(D, self.n_clusters)
        centers = [segments[i] for i in medoid_indices]
        self.cluster_labels_ = labels
        self.cluster_centers_ = centers
        return labels, centers

    def classify_shape(self, segment, cluster_centers):
        if cluster_centers is None:
            return -1, np.inf
        best_idx = -1
        best_dist = np.inf
        for i, c in enumerate(cluster_centers):
            _, dist = self._procrustes_align(c, segment)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx, best_dist

    def assign_shape_names(self, labels, segments):
        """
        Heuristic naming based on curvature and slopes – relaxed for better differentiation.
        """
        names = []
        for seg in segments:
            y = seg[:, 1]
            d2 = np.gradient(np.gradient(y))
            mid = len(d2)//2
            curv = abs(d2[mid])
            half = len(y)//2
            slope1 = (y[half] - y[0]) / half if half > 0 else 0
            slope2 = (y[-1] - y[half]) / (len(y)-half) if (len(y)-half)>0 else 0
            # Relaxed thresholds
            if slope1 < -0.3 and slope2 > 0.3:
                name = "V"
            elif curv < 0.8 and abs(slope1) < 0.1 and abs(slope2) < 0.1:
                name = "U"
            else:
                name = "L"
            names.append(name)
        cluster_name_map = {}
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) == 0:
                cluster_name_map[k] = "unknown"
            else:
                cluster_names = [names[i] for i, m in enumerate(mask) if m]
                most_common = Counter(cluster_names).most_common(1)[0][0]
                cluster_name_map[k] = most_common
        return cluster_name_map

    def analyze(self, prices):
        segments = self.find_recovery_segments(prices)
        if len(segments) < self.n_clusters:
            return {"error": "insufficient segments", "num_recoveries": len(segments)}, None, None
        labels, centers = self.cluster_shapes(segments)
        name_map = self.assign_shape_names(labels, segments)
        return {
            "segments": segments,
            "labels": labels,
            "cluster_names": name_map,
            "num_recoveries": len(segments)
        }, centers, labels
