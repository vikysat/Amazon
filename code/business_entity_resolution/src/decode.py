"""Decoding: turn pair probabilities into per-S1 match sets.

Operations (all vectorised over pair tables with integer indices i1 / ip):
  * one_to_one:     keep each pool record only for its highest-probability S1
                    (EDA: every S2/S3 record belongs to at most one S1).
  * threshold:      keep pairs with p >= t.
  * expected-F:     per S1 choose top-k maximising Monte-Carlo expected F0.5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import fbeta_counts


def one_to_one(i1: np.ndarray, ip: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Boolean mask selecting, for every pool record ``ip``, only its best-``p`` pair."""
    order = np.lexsort((-p, ip))
    first = np.ones(len(order), dtype=bool)
    first[1:] = ip[order][1:] != ip[order][:-1]
    mask = np.zeros(len(p), dtype=bool)
    mask[order[first]] = True
    return mask


def score_int(eval_s1: np.ndarray, pred_i1: np.ndarray, pred_ip: np.ndarray,
              true_i1: np.ndarray, true_ip: np.ndarray, n_s1: int) -> dict:
    """Macro F0.5 over the S1 row indices ``eval_s1`` from integer pair arrays."""
    ev = np.zeros(n_s1, dtype=bool)
    ev[eval_s1] = True
    pm = ev[pred_i1]
    tm = ev[true_i1]
    pi1, pip = pred_i1[pm], pred_ip[pm]
    ti1, tip = true_i1[tm], true_ip[tm]
    n_pred = np.bincount(pi1, minlength=n_s1)
    n_true = np.bincount(ti1, minlength=n_s1)
    kp = pi1.astype(np.int64) * 20_000_000 + pip
    kt = ti1.astype(np.int64) * 20_000_000 + tip
    hit = np.isin(kp, kt)
    tp = np.bincount(pi1[hit], minlength=n_s1)
    f = fbeta_counts(tp[ev], n_pred[ev], n_true[ev])
    single = n_true[ev] == 0
    return {"f": float(f.mean()), "singleton_acc": float(f[single].mean()) if single.any() else np.nan,
            "nonsingleton_f": float(f[~single].mean()), "n": int(ev.sum())}


def tune_threshold(eval_s1, i1, ip, p, true_i1, true_ip, n_s1, grid=None, o2o: bool = True):
    """Grid-search a global probability threshold (after optional one-to-one). Returns (best_t, table)."""
    grid = np.round(np.arange(0.05, 0.96, 0.025), 3) if grid is None else grid
    mask = one_to_one(i1, ip, p) if o2o else np.ones(len(p), bool)
    rows = []
    for t in grid:
        m = mask & (p >= t)
        r = score_int(eval_s1, i1[m], ip[m], true_i1, true_ip, n_s1)
        r["t"] = float(t)
        rows.append(r)
    tab = pd.DataFrame(rows)
    return float(tab.loc[tab["f"].idxmax(), "t"]), tab


def expected_f_decode(i1: np.ndarray, p: np.ndarray, n_s1: int, max_k: int = 10,
                      n_mc: int = 500, missed_true: np.ndarray | None = None, seed: int = 42) -> np.ndarray:
    """Choose, per S1, the top-k (k=0..max_k) set maximising expected F0.5 (Monte Carlo).

    ``missed_true`` (per-S1 expected count of true matches not among candidates, from OOF)
    is added to |T| in the simulation. Returns a boolean mask over the input pairs.
    """
    rng = np.random.default_rng(seed)
    order = np.lexsort((-p, i1))
    i1s, ps = i1[order], p[order]
    starts = np.flatnonzero(np.r_[True, i1s[1:] != i1s[:-1]])
    lens = np.diff(np.r_[starts, len(i1s)])
    rank = np.arange(len(i1s)) - np.repeat(starts, lens)
    keep_sorted = np.zeros(len(i1s), dtype=bool)
    K = max_k
    # dense matrix of top-K probs per S1 group
    g = np.repeat(np.arange(len(starts)), lens)
    sel = rank < K
    P = np.zeros((len(starts), K), dtype=np.float32)
    P[g[sel], rank[sel]] = ps[sel]
    extra = np.zeros(len(starts)) if missed_true is None else missed_true[i1s[starts]]
    best_k = np.zeros(len(starts), dtype=np.int64)
    # process in blocks to bound memory: B x n_mc x K
    B = 20000
    for b0 in range(0, len(starts), B):
        Pb = P[b0:b0 + B]
        u = rng.random((Pb.shape[0], n_mc, K), dtype=np.float32)
        Y = (u < Pb[:, None, :])  # sampled truth of each ranked candidate
        n_true = Y.sum(2) + np.round(extra[b0:b0 + B])[:, None]
        cum_tp = np.cumsum(Y, axis=2)  # tp when predicting top-k (k=1..K)
        ef = np.zeros((Pb.shape[0], K + 1))
        ef[:, 0] = (n_true == 0).mean(1)
        for k in range(1, K + 1):
            f = fbeta_counts(cum_tp[:, :, k - 1].ravel(), np.full(cum_tp[:, :, 0].size, k),
                             n_true.ravel()).reshape(n_true.shape)
            ef[:, k] = f.mean(1)
        # candidates beyond the group's size are invalid
        kmax = np.minimum(lens[b0:b0 + B], K)
        invalid = np.arange(K + 1)[None, :] > kmax[:, None]
        ef[invalid] = -1
        best_k[b0:b0 + B] = ef.argmax(1)
    keep_sorted = rank < np.repeat(best_k, lens)
    mask = np.zeros(len(p), dtype=bool)
    mask[order[keep_sorted]] = True
    return mask
