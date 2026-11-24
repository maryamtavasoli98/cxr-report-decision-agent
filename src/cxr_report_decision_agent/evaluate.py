"""Metrics written from scratch: binary P/R/F1, risk-coverage curves, bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np


def prf(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp, fp, fn = int((y & pred).sum()), int((~y & pred).sum()), int((y & ~pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def risk_coverage(
    conf: np.ndarray, correct: np.ndarray, y: np.ndarray, pred: np.ndarray, grid: np.ndarray
) -> list[dict]:
    """Answer the most confident fraction `c` of pairs, abstain on the rest. Reports error rate and positive-class
    micro F1 over the answered pairs, and the fraction of true positives that were abstained on."""
    order = np.argsort(-conf, kind="stable")
    n, npos = len(conf), int(np.asarray(y).sum())
    rows = []
    for c in grid:
        k = max(1, int(round(c * n)))
        idx = order[:k]
        m = prf(y[idx], pred[idx])
        rows.append(
            {
                "coverage": float(c),
                "error_rate": float(1 - correct[idx].mean()),
                "micro_f1": m["f1"],
                "positives_abstained": float(1 - y[idx].sum() / npos) if npos else 0.0,
            }
        )
    return rows


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better): mean selective error over all coverage levels."""
    order = np.argsort(-conf, kind="stable")
    err = 1 - correct[order]
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def bootstrap_ci(stat, n_items: int, n_boot: int, seed: int = 0) -> tuple[float, float, float]:
    """`stat(idx)` computes the statistic on resampled item indices. Returns (point, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    point = stat(np.arange(n_items))
    vals = [stat(rng.integers(0, n_items, n_items)) for _ in range(n_boot)]
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mean_std(xs) -> tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0
