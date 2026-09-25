"""Test-like validation proxy.

Motivation: the Phase-4 baseline scored 0.7335 on train but 0.645 on the public leaderboard.
Test differs from train in composition (more pool records per S1, likely more singletons and
unowned distractors, plus the unseen country). This module builds a *proxy* validation set from
train OOF predictions by simulating that composition, without any test labels:

  * singleton injection: for a fraction ``a`` of S1 entities, all their true pool records are
    removed (their pairs are dropped everywhere) -> the entity becomes a singleton whose
    remaining candidates are distractors;
  * distractor injection: a fraction ``b`` of S1 entities is removed; their true pool records
    stay in the pool (their pairs with *other* S1s are kept) -> unowned distractors.

The proxy is applied to already-scored pair tables (i1, ip, p), so it is cheap. ``a``/``b`` are
chosen so that label-free statistics of the proxy (candidates per S1, share of S1s whose best
candidate is weak, pool/S1 ratio) match the same statistics on test. Removing records does not
re-run blocking; this approximates the fact that removed records would free up top-k slots.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from decode import one_to_one, score_int


def label_free_stats(i1: np.ndarray, ip: np.ndarray, cos: np.ndarray, n_s1: int, n_pool: int) -> dict:
    """Composition statistics computable on train and test alike (no labels)."""
    cps = np.bincount(i1, minlength=n_s1)
    best = np.full(n_s1, 0.0)
    np.maximum.at(best, i1, cos)
    n_owner = np.bincount(ip, minlength=n_pool)
    return {
        "cands_per_s1": float(cps.mean()),
        "s1_no_cand": float((cps == 0).mean()),
        "best_cos_q10": float(np.quantile(best, 0.10)),
        "best_cos_q25": float(np.quantile(best, 0.25)),
        "best_cos_median": float(np.median(best)),
        "share_best_cos_lt_0.4": float((best < 0.4).mean()),
        "pool_with_cand_share": float((n_owner > 0).mean()),
        "s1_per_cand_mean": float(n_owner[n_owner > 0].mean()),
    }


def make_proxy(i1, ip, true_i1, true_ip, n_s1: int, a: float, b: float, seed: int = 0):
    """Return (pair_mask, eval_s1, true_mask) implementing singleton/distractor injection.

    pair_mask selects the pairs that survive; eval_s1 are the S1 rows evaluated (removed S1s
    excluded); true_mask selects surviving ground-truth pairs.
    """
    rng = np.random.default_rng(seed)
    u = rng.random(n_s1)
    make_single = u < a                 # these S1s lose their true pool records
    removed = (u >= a) & (u < a + b)    # these S1s disappear (their records become distractors)
    dead_pool = np.zeros(max(ip.max(), true_ip.max()) + 1, dtype=bool)
    dead_pool[true_ip[make_single[true_i1]]] = True
    pair_mask = ~removed[i1] & ~dead_pool[ip]
    true_mask = ~removed[true_i1] & ~dead_pool[true_ip]
    eval_s1 = np.flatnonzero(~removed)
    return pair_mask, eval_s1, true_mask


def proxy_score(i1, ip, p, true_i1, true_ip, n_s1, a, b, t, seed: int = 0) -> dict:
    """Score one-to-one + threshold ``t`` decoding on the proxy defined by (a, b)."""
    m, ev, tm = make_proxy(i1, ip, true_i1, true_ip, n_s1, a, b, seed)
    pi1, pip, pp = i1[m], ip[m], p[m]
    keep = one_to_one(pi1, pip, pp) & (pp >= t)
    r = score_int(ev, pi1[keep], pip[keep], true_i1[tm], true_ip[tm], n_s1)
    r["singleton_rate"] = float(np.isin(ev, true_i1[tm], invert=True).mean())
    return r
