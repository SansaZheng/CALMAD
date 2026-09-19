"""Detector pool with fixed default hyperparameters (never tuned per dataset).

Every detector supports:
    fit(X)            -- fit on (possibly contaminated) inlier data
    score(X) -> s     -- higher = more anomalous, works on unseen points
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf


class KNNDetector:
    """Mean distance to the k nearest neighbors in the fit set."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X):
        self.k_eff_ = min(self.k, max(1, X.shape[0] - 1))
        self.nn_ = NearestNeighbors(n_neighbors=self.k_eff_).fit(X)
        return self

    def score(self, X):
        d, _ = self.nn_.kneighbors(X, n_neighbors=self.k_eff_)
        return d.mean(axis=1)


class LOFDetector:
    def __init__(self, k=20):
        self.k = k

    def fit(self, X):
        k = min(self.k, max(2, X.shape[0] - 1))
        self.lof_ = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(X)
        return self

    def score(self, X):
        return -self.lof_.score_samples(X)


class IForestDetector:
    def __init__(self, n_estimators=200, max_samples="auto", seed=0):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.seed = seed

    def fit(self, X):
        ms = self.max_samples
        if isinstance(ms, int):
            ms = min(ms, X.shape[0])
        self.f_ = IsolationForest(
            n_estimators=self.n_estimators, max_samples=ms,
            random_state=self.seed,
        ).fit(X)
        return self

    def score(self, X):
        return -self.f_.score_samples(X)


class PCADetector:
    """Reconstruction error using components explaining 90% of variance."""

    def __init__(self, var=0.9):
        self.var = var

    def fit(self, X):
        n_max = min(X.shape[0], X.shape[1])
        p = PCA(n_components=n_max).fit(X)
        cum = np.cumsum(p.explained_variance_ratio_)
        k = int(np.searchsorted(cum, self.var) + 1)
        k = max(1, min(k, n_max - 1)) if n_max > 1 else 1
        self.pca_ = PCA(n_components=k).fit(X)
        return self

    def score(self, X):
        Z = self.pca_.inverse_transform(self.pca_.transform(X))
        return np.sqrt(((X - Z) ** 2).sum(axis=1))


class MahalanobisDetector:
    """Mahalanobis distance under a Ledoit-Wolf shrinkage covariance."""

    def fit(self, X):
        self.lw_ = LedoitWolf().fit(X)
        return self

    def score(self, X):
        return self.lw_.mahalanobis(X)


class HBOSDetector:
    """Histogram-based outlier score with fixed 10 bins per feature."""

    def __init__(self, bins=10):
        self.bins = bins

    def fit(self, X):
        self.edges_, self.dens_ = [], []
        n = X.shape[0]
        for j in range(X.shape[1]):
            h, e = np.histogram(X[:, j], bins=self.bins)
            d = h / max(n, 1)
            self.edges_.append(e)
            self.dens_.append(np.maximum(d, 1e-12))
        return self

    def score(self, X):
        s = np.zeros(X.shape[0])
        for j in range(X.shape[1]):
            e, d = self.edges_[j], self.dens_[j]
            idx = np.clip(np.searchsorted(e, X[:, j], side="right") - 1, 0, len(d) - 1)
            dj = d[idx].copy()
            # points outside the fit range get the minimum observed density
            out = (X[:, j] < e[0]) | (X[:, j] > e[-1])
            dj[out] = d.min()
            s += -np.log(dj)
        return s


class ECODDetector:
    """Empirical-CDF based detector (Li et al., 2022), hyperparameter-free."""

    def fit(self, X):
        self.X_ = np.sort(X, axis=0)
        self.n_ = X.shape[0]
        self.skew_ = self._skew(X)
        return self

    @staticmethod
    def _skew(X):
        mu = X.mean(axis=0)
        sd = X.std(axis=0) + 1e-12
        return (((X - mu) / sd) ** 3).mean(axis=0)

    def _ecdf(self, X):
        # P(train <= x) with a +1 correction, per feature
        n = self.n_
        left = np.empty_like(X, dtype=float)
        for j in range(X.shape[1]):
            r = np.searchsorted(self.X_[:, j], X[:, j], side="right")
            left[:, j] = (r + 1) / (n + 2)
        return left

    def score(self, X):
        Fl = self._ecdf(X)          # left tail prob
        Fr = 1.0 - Fl + 2.0 / (self.n_ + 2)  # right tail prob (with correction)
        Fl = np.clip(Fl, 1e-12, 1)
        Fr = np.clip(Fr, 1e-12, 1)
        Ol = -np.log(Fl).sum(axis=1)
        Or = -np.log(Fr).sum(axis=1)
        Oa = np.where(self.skew_ < 0, -np.log(Fl), -np.log(Fr)).sum(axis=1)
        return np.maximum.reduce([Ol, Or, Oa])


# Pool balanced across mechanism families so that no single failure mode
# (e.g. distance-based detectors inverted by a dense contamination cluster)
# can command a majority of the consensus:
#   distance: knn25, lof20 | isolation: iforest, iforest1k
#   density: hbos, ecod    | reconstruction: pca | covariance: maha
DEFAULT_POOL = [
    ("knn25", lambda seed: KNNDetector(k=25)),
    ("lof20", lambda seed: LOFDetector(k=20)),
    ("iforest", lambda seed: IForestDetector(seed=seed)),
    ("iforest1k", lambda seed: IForestDetector(max_samples=1024, seed=seed + 1)),
    ("pca", lambda seed: PCADetector()),
    ("maha", lambda seed: MahalanobisDetector()),
    ("hbos", lambda seed: HBOSDetector()),
    ("ecod", lambda seed: ECODDetector()),
]

def make_pool(seed=0):
    return [(name, ctor(seed)) for name, ctor in DEFAULT_POOL]
