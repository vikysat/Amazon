"""Competition metric: macro F-beta (beta=0.5) averaged over Source-1 entities.

Per S1 entity with predicted set P and true set T:
  * T empty and P empty            -> 1.0
  * exactly one of T / P empty     -> 0.0
  * otherwise F = (1+b^2) p r / (b^2 p + r), 0 if |P & T| == 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BETA = 0.5


def fbeta_sets(pred: set, true: set, beta: float = BETA) -> float:
    """Score one S1 entity given predicted and true id sets."""
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def fbeta_counts(tp: np.ndarray, n_pred: np.ndarray, n_true: np.ndarray, beta: float = BETA) -> np.ndarray:
    """Vectorised per-entity F-beta from counts (tp, |P|, |T|) arrays."""
    tp, n_pred, n_true = (np.asarray(x, dtype=np.float64) for x in (tp, n_pred, n_true))
    b2 = beta * beta
    # F = (1+b2) tp / (b2 |T| + |P|)  (algebraically equal to the p/r form)
    denom = b2 * n_true + n_pred
    f = np.where(denom > 0, (1 + b2) * tp / np.maximum(denom, 1e-12), 0.0)
    f = np.where((n_pred == 0) & (n_true == 0), 1.0, f)
    return f


def score_pairs(s1_ids, pred_pairs: pd.DataFrame, true_pairs: pd.DataFrame, beta: float = BETA) -> dict:
    """Macro F-beta over ``s1_ids`` from long pair tables with columns (s1, cand).

    Returns a dict with overall score, singleton accuracy (on truly-empty
    entities), non-singleton mean F, and counts.
    """
    s1_ids = pd.Index(pd.unique(np.asarray(s1_ids, dtype=object)))
    pp = pred_pairs[["s1", "cand"]].drop_duplicates()
    tt = true_pairs[["s1", "cand"]].drop_duplicates()
    pp = pp[pp["s1"].isin(s1_ids)]
    tt = tt[tt["s1"].isin(s1_ids)]
    n_pred = pp.groupby("s1").size().reindex(s1_ids, fill_value=0).to_numpy()
    n_true = tt.groupby("s1").size().reindex(s1_ids, fill_value=0).to_numpy()
    inter = pp.merge(tt, on=["s1", "cand"])
    tp = inter.groupby("s1").size().reindex(s1_ids, fill_value=0).to_numpy()
    f = fbeta_counts(tp, n_pred, n_true, beta)
    single = n_true == 0
    return {
        "f": float(f.mean()),
        "singleton_acc": float(f[single].mean()) if single.any() else float("nan"),
        "nonsingleton_f": float(f[~single].mean()) if (~single).any() else float("nan"),
        "n": int(len(s1_ids)),
        "singleton_rate": float(single.mean()),
    }


def _self_test() -> None:
    """Unit test against the worked example of the problem statement."""
    assert abs(fbeta_sets({"A", "B", "C"}, {"A", "C"}) - 0.714) < 1e-3
    assert fbeta_sets(set(), set()) == 1.0
    assert fbeta_sets({"A"}, set()) == 0.0
    assert fbeta_sets(set(), {"A"}) == 0.0
    assert fbeta_sets({"B"}, {"A"}) == 0.0
    v = fbeta_counts([2], [3], [2])[0]
    assert abs(v - fbeta_sets({"A", "B", "C"}, {"A", "C"})) < 1e-12
    pred = pd.DataFrame({"s1": ["x", "x", "x", "y"], "cand": ["A", "B", "C", "Q"]})
    true = pd.DataFrame({"s1": ["x", "x"], "cand": ["A", "C"]})
    r = score_pairs(["x", "y", "z"], pred, true)
    assert abs(r["f"] - (fbeta_sets({"A", "B", "C"}, {"A", "C"}) + 0 + 1) / 3) < 1e-12
    print("metrics self-test OK:", round(fbeta_sets({"A", "B", "C"}, {"A", "C"}), 4))


if __name__ == "__main__":
    _self_test()
