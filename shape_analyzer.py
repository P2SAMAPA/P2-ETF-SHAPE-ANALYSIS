"""
Shape analysis using Kendall's shape space and Procrustes distance.
Detects recovery segments, aligns them, classifies into V/U/L shapes.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from collections import defaultdict

class ShapeAnalyzer:
    def __init__(self, trough_window=5, peak_threshold=0.10, min_recovery_days=5,
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
        """
        prices: pd.Series indexed by datetime.
        Returns list of segments, each segment is a 2D array of (normalized time, normalized price).
        """
        segments = []
        dates = prices.index
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

        # For each trough, find subsequent peak with sufficient gain
        for ti in trough_idx:
            trough_price = values[ti]
            # look forward up to 1 year (252 days) for a peak
            max_lookahead = min(n, ti + 252)
            peak_candidate = None
            for j in range(ti+1, max_lookahead):
                if values[j] >= trough_price * (1 + self.peak_threshold):
                    peak_candidate = j
                    break
            if peak_candidate is None:
                continue
            # Extract segment from trough to peak
            segment_prices = values[ti:peak_candidate+1]
            if len(segment_prices) < self.min_recovery_days:
                continue
            # Normalise time to [0,1]
            time_norm = np.linspace(0, 1, len(segment_prices))
            # Normalise price to [0,1]
            price_min = segment_prices.min()
            price_max = segment_prices.max()
            if price_max == price_min:
                continue
            price_norm = (segment_prices - price_min) / (price_max - price_min)
            # Build 2D points: x = time, y = price
            segment = np.column_stack([time_norm, price_norm])
            # Interpolate to fixed number of points
            interp_time = np.linspace(0, 1, self.interp_points)
            f_x = interp1d(time_norm, segment_prices, kind='linear', fill_value='extrapolate')
            f_y = interp1d(time_norm, price_norm, kind='linear', fill_value='extrapolate')
            interp_prices = f_x(interp_time)
            interp_norm = f_y(interp_time)
            # Re-normalise after interpolation (optional, but ensures [0,1] range)
            interp_norm = (interp_norm - interp_norm.min()) / (interp_norm.max() - interp_norm.min() + 1e-8)
            interp_segment = np.column_stack([interp_time, interp_norm])
            segments.append(interp_segment)
        return segments

    def _procrustes_align(self, X, Y, max_iter=10):
        """
        Procrustes superimposition: translate, scale, rotate Y to best fit X.
        Returns aligned Y and the sum of squared distances (Procrustes distance).
        """
        # Center both
        X_cent = X - X.mean(axis=0)
        Y_cent = Y - Y.mean(axis=0)
        # Scale to centroid size = 1
        size_X = np.sqrt(np.sum(X_cent**2))
        size_Y = np.sqrt(np.sum(Y_cent**2))
        if size_X < 1e-8 or size_Y < 1e-8:
            return Y, np.inf
        X_scaled = X_cent / size_X
        Y_scaled = Y_cent / size_Y
        # Orthogonal rotation (Procrustes rotation)
        M = Y_scaled.T @ X_scaled
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        Y_rotated = Y_scaled @ R
        # Compute distance (sum of squared differences)
        dist = np.sum((X_scaled - Y_rotated)**2)
        return Y_rotated, dist

    def compute_procrustes_matrix(self, segments):
        """Return pairwise Procrustes distances between all segments."""
        n = len(segments)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                _, dist = self._procrustes_align(segments[i], segments[j])
                D[i,j] = D[j,i] = dist
        return D

    def cluster_shapes(self, segments):
        """Cluster segments into n_clusters using k-medoids on Procrustes distances."""
        if len(segments) < self.n_clusters:
            return None, None
        from sklearn_extra.cluster import KMedoids
        D = self.compute_procrustes_matrix(segments)
        kmed = KMedoids(n_clusters=self.n_clusters, metric='precomputed', random_state=42)
        labels = kmed.fit_predict(D)
        # Compute cluster centers (medoids)
        centers = []
        for k in range(self.n_clusters):
            idx = np.where(labels == k)[0][0]   # medoid index
            centers.append(segments[idx])
        self.cluster_labels_ = labels
        self.cluster_centers_ = centers
        return labels, centers

    def classify_shape(self, segment, cluster_centers):
        """Return closest cluster index and Procrustes distance."""
        if cluster_centers is None:
            return -1, np.inf
        best_idx = -1
        best_dist = np.inf
        for i, center in enumerate(cluster_centers):
            _, dist = self._procrustes_align(center, segment)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx, best_dist

    def assign_shape_names(self, labels, segments):
        """
        Heuristic naming: compute curvature (second derivative) at midpoint.
        V shape: negative then positive (sharp turn) – high second derivative absolute.
        U shape: flat bottom (low curvature).
        L shape: steep initial drop then flat (asymmetric).
        """
        names = []
        for i, seg in enumerate(segments):
            # second derivative via finite differences
            y = seg[:,1]
            d2 = np.gradient(np.gradient(y))
            # curvature at midpoint: absolute value of second derivative
            mid = len(d2)//2
            curv = abs(d2[mid])
            # slope in first half vs second half
            half = len(y)//2
            slope1 = (y[half] - y[0]) / half
            slope2 = (y[-1] - y[half]) / half
            if slope1 < -0.5 and slope2 > 0.5:
                name = "V"
            elif abs(slope1) < 0.2 and abs(slope2) < 0.2:
                name = "L"  # flat, but not U
            elif curv < 0.5:
                name = "U"
            else:
                name = "mixed"
            names.append(name)
        # Rename clusters based on majority vote
        cluster_name_map = {}
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) == 0:
                cluster_name_map[k] = "unknown"
            else:
                cluster_names = [names[i] for i, m in enumerate(mask) if m]
                from collections import Counter
                most_common = Counter(cluster_names).most_common(1)[0][0]
                cluster_name_map[k] = most_common
        return cluster_name_map

    def analyze(self, prices):
        """Full pipeline: find segments, cluster, return cluster names and centroids."""
        segments = self.find_recovery_segments(prices)
        if len(segments) < self.n_clusters:
            return {"error": "insufficient segments"}, None, None
        labels, centers = self.cluster_shapes(segments)
        name_map = self.assign_shape_names(labels, segments)
        return {"segments": segments, "labels": labels, "cluster_names": name_map}, centers, labels
