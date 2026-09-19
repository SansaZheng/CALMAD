"""Baseline detectors trained on the full (contaminated) training pool with
their standard sklearn defaults, plus a 'naive in-sample calibration'
comparator that mirrors common practice (no held-out calibration split).
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

from .detectors import ECODDetector, PCADetector


def _subsample(X, n_max, seed):
    if X.shape[0] <= n_max:
        return X
    rng = np.random.default_rng(seed)
    return X[rng.choice(X.shape[0], n_max, replace=False)]


def run_baselines(X_train, X_test, seed=0, gamma=None):
    """Returns {name: test_scores} for standard baselines.

    When gamma is given, three contamination-aware deep baselines are added:
    LOE with the oracle ratio alpha0 = gamma, LOE with a fixed misspecified
    alpha0 = 0.10, and blind Deep SVDD (alpha0 = 0)."""
    sc = StandardScaler().fit(X_train)
    Xtr = sc.transform(X_train)
    Xte = sc.transform(X_test)
    out = {}

    f = IsolationForest(n_estimators=200, random_state=seed).fit(Xtr)
    out["iforest"] = -f.score_samples(Xte)

    k = min(20, Xtr.shape[0] - 1)
    lof = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(Xtr)
    out["lof"] = -lof.score_samples(Xte)

    nn = NearestNeighbors(n_neighbors=min(5, Xtr.shape[0] - 1)).fit(Xtr)
    out["knn"] = nn.kneighbors(Xte)[0].mean(axis=1)

    Xsub = _subsample(Xtr, 8000, seed)
    oc = OneClassSVM(nu=0.5, gamma="scale").fit(Xsub)
    out["ocsvm"] = -oc.score_samples(Xte)

    p = PCADetector().fit(Xtr)
    out["pca"] = p.score(Xte)

    e = ECODDetector().fit(Xtr)
    out["ecod"] = e.score(Xte)

    if gamma is not None:
        from .loe import LOEDetector, DeepSVDDDetector
        out["loe"] = LOEDetector(alpha0=gamma, mode="hard", seed=seed).fit(Xtr).score(Xte)
        out["loe_soft"] = LOEDetector(alpha0=gamma, mode="soft", seed=seed).fit(Xtr).score(Xte)
        out["loe_mis"] = LOEDetector(alpha0=0.10, mode="hard", seed=seed).fit(Xtr).score(Xte)
        out["dsvdd"] = DeepSVDDDetector(seed=seed).fit(Xtr).score(Xte)
    return out


def loe_conformal(X_train, X_test, alpha0, seed=0, cal_frac=0.4):
    """LOE used as the score function inside the same split-conformal protocol
    as CALM-AD: identical 60/40 split and calibration rule, no trimming, so the
    only difference is the score. Returns (test_scores, cal_scores)."""
    import numpy as np
    from .loe import LOEDetector
    from .pipeline import RobustScaler_
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    perm = rng.permutation(n)
    n_cal = int(round(cal_frac * n))
    cal_idx, fit_idx = perm[:n_cal], perm[n_cal:]
    sc = RobustScaler_().fit(X_train[fit_idx])
    det = LOEDetector(alpha0=alpha0, mode="hard", seed=seed).fit(sc.transform(X_train[fit_idx]))
    return det.score(sc.transform(X_test)), det.score(sc.transform(X_train[cal_idx]))


def naive_knn_scores(X_train, X_test, k=5, seed=0):
    """kNN detector with FAIR in-sample scoring (self excluded) -- the classic
    memorization case for naive in-sample calibration."""
    sc = StandardScaler().fit(X_train)
    Xtr, Xte = sc.transform(X_train), sc.transform(X_test)
    k = min(k, Xtr.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xtr)
    d_tr, _ = nn.kneighbors(Xtr)
    s_tr = d_tr[:, 1:].mean(axis=1)  # drop self
    d_te, _ = nn.kneighbors(Xte, n_neighbors=k)
    s_te = d_te.mean(axis=1)
    return s_tr, s_te


def naive_insample_pvalues(train_scores, test_scores):
    """The common (invalid) shortcut: calibrate on the same data the detector
    was trained on. p = (1 + #{train >= s}) / (n_train + 1)."""
    tr = np.sort(np.asarray(train_scores, dtype=float))
    s = np.asarray(test_scores, dtype=float)
    n = len(tr)
    ge = n - np.searchsorted(tr, s, side="left")
    return (1.0 + ge) / (n + 1.0)
