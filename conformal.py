"""Split-conformal p-values, BH-based FDR control, and weighted conformal
p-values for covariate shift.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def conformal_pvalues(cal_scores, test_scores):
    """p_i = (1 + #{cal >= s_i}) / (n_cal + 1). Higher score = more anomalous."""
    cal = np.sort(np.asarray(cal_scores, dtype=float))
    s = np.asarray(test_scores, dtype=float)
    n = len(cal)
    # number of cal scores >= s  ==  n - #(cal < s)
    ge = n - np.searchsorted(cal, s, side="left")
    return (1.0 + ge) / (n + 1.0)


def bh_reject(pvals, q):
    """Benjamini-Hochberg: returns boolean rejection mask at FDR level q."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    if not passed.any():
        return np.zeros(m, dtype=bool)
    k = np.max(np.nonzero(passed)[0])
    cut = p[order][k]
    return p <= cut


def estimate_shift_weights(X_cal, X_test, clip=50.0, seed=0):
    """Likelihood-ratio weights w(x) ~ p_test(x)/p_cal(x) from a regularized
    logistic discriminator trained on covariates only (never on scores/labels).
    """
    Xa = np.vstack([X_cal, X_test])
    ya = np.concatenate([np.zeros(len(X_cal)), np.ones(len(X_test))])
    sc = StandardScaler().fit(Xa)
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(
        sc.transform(Xa), ya
    )
    n0, n1 = len(X_cal), len(X_test)

    def w(X):
        logit = clf.decision_function(sc.transform(X))
        # odds ratio, corrected for class imbalance in the discriminator
        lr = np.exp(logit) * (n0 / n1)
        return np.clip(lr, 1.0 / clip, clip)

    return w


def weighted_conformal_pvalues(cal_scores, test_scores, w_cal, w_test):
    """Weighted conformal p-values (Tibshirani et al., 2019 style):
    p(x) = [sum_i w_i 1{s_i >= s(x)} + w(x)] / [sum_i w_i + w(x)]
    """
    cal = np.asarray(cal_scores, dtype=float)
    s = np.asarray(test_scores, dtype=float)
    order = np.argsort(cal)
    cal_sorted = cal[order]
    w_sorted = np.asarray(w_cal, dtype=float)[order]
    # suffix sums of weights for cal scores >= s
    suffix = np.concatenate([np.cumsum(w_sorted[::-1])[::-1], [0.0]])
    idx = np.searchsorted(cal_sorted, s, side="left")
    wsum_ge = suffix[idx]
    total = w_sorted.sum()
    wt = np.asarray(w_test, dtype=float)
    return (wsum_ge + wt) / (total + wt)
