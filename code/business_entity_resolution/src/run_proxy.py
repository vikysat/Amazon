"""Calibrate the test-like proxy (src/proxy_val.py) and use it to choose the decoding.

Steps:
  1. label-free composition stats on train vs test (per country, and overall);
  2. grid over (a = singleton-injection rate, b = S1-removal rate) on train, choosing the pair whose
     proxy stats best match test (US+India only, since train has no France);
  3. sanity check: baseline score on that proxy should be close to the public LB (0.645);
  4. score stage-2 OOF decodings (threshold grid, expected-F) on the proxy; report the best.

Usage: python src/run_proxy.py --cache-dir cache
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from decode import expected_f_decode, one_to_one, score_int  # noqa: E402
from io_utils import log  # noqa: E402
from prepare import load_prepared  # noqa: E402
from proxy_val import label_free_stats, make_proxy, proxy_score  # noqa: E402
from stages import feats_path, label_pairs, stage_block, stage_filter  # noqa: E402

KEYS = ["cands_per_s1", "share_best_cos_lt_0.4", "best_cos_q10", "best_cos_q25", "s1_per_cand_mean"]


def stats_by_country(pairs, countries, n_pool, mask=None, eval_s1=None):
    """label_free_stats per country (restricted to eval_s1 rows / surviving pairs if given)."""
    out = {}
    i1, ip, cos = pairs["i1"].to_numpy(), pairs["ip"].to_numpy(), pairs["cos"].to_numpy()
    if mask is not None:
        i1, ip, cos = i1[mask], ip[mask], cos[mask]
    keep_s1 = np.ones(len(countries), bool) if eval_s1 is None else np.isin(np.arange(len(countries)), eval_s1)
    for c in np.unique(countries):
        sel_s1 = (countries == c) & keep_s1
        remap = -np.ones(len(countries), np.int64)
        remap[sel_s1] = np.arange(sel_s1.sum())
        m = remap[i1] >= 0
        out[c] = label_free_stats(remap[i1[m]], ip[m], cos[m], int(sel_s1.sum()), n_pool)
    return out


def dist(a: dict, b: dict) -> float:
    """Relative distance between two stat dicts over KEYS."""
    return float(sum(abs(a[k] - b[k]) / max(abs(b[k]), 1e-3) for k in KEYS))


def main() -> None:
    """Run the proxy calibration and decoding comparison."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    a = ap.parse_args()
    cache = a.cache_dir
    tr = stage_filter(stage_block(cache, "train"))
    y, true = label_pairs(cache, tr)
    c_tr = load_prepared(cache, "train", "s1", ["country"])["country"].astype(str).to_numpy()
    te = stage_filter(stage_block(cache, "test"))
    c_te = load_prepared(cache, "test", "s1", ["country"])["country"].astype(str).to_numpy()
    n_tr, n_te = len(c_tr), len(c_te)
    st_tr = stats_by_country(tr, c_tr, 10_320_219)
    st_te = stats_by_country(te, c_te, 9_969_589)
    log("train stats: " + json.dumps(st_tr, indent=1))
    log("test stats: " + json.dumps(st_te, indent=1))

    i1, ip = tr["i1"].to_numpy(), tr["ip"].to_numpy()
    ti1, tip = true["i1"].to_numpy(), true["ip"].to_numpy()
    grid = []
    for sa in (0.0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25):
        for sb in (0.0, 0.1, 0.2, 0.3):
            m, ev, _ = make_proxy(i1, ip, ti1, tip, n_tr, sa, sb)
            st = stats_by_country(tr, c_tr, 10_320_219, mask=m, eval_s1=ev)
            d = sum(dist(st[c], st_te[c]) for c in ("US", "India"))
            grid.append((d, sa, sb))
            log(f"  a={sa} b={sb} dist={d:.3f} US={ {k: round(st['US'][k], 3) for k in KEYS} }")
    grid.sort()
    d, sa, sb = grid[0]
    log(f"best proxy: a={sa} b={sb} dist={d:.3f}")

    # baseline on proxy (compare with LB 0.645)
    Xb = pd.read_parquet(feats_path(cache, "train", tr), columns=["n_tset", "a_tset"])
    pb = ((Xb["n_tset"].to_numpy() + Xb["a_tset"].to_numpy()) / 200).astype(np.float32)
    res = {"a": sa, "b": sb, "dist": d}
    res["baseline_train"] = proxy_score(i1, ip, pb, ti1, tip, n_tr, 0.0, 0.0, 0.8)
    res["baseline_proxy"] = proxy_score(i1, ip, pb, ti1, tip, n_tr, sa, sb, 0.8)
    log(f"baseline: train {res['baseline_train']['f']:.4f} proxy {res['baseline_proxy']['f']:.4f} (LB 0.645)")

    # stage-2 OOF decodings on proxy
    p2 = np.load(Path(cache) / "oof_stage2.npy")
    rows = []
    for t in np.round(np.arange(0.4, 0.96, 0.05), 2):
        r = proxy_score(i1, ip, p2, ti1, tip, n_tr, sa, sb, t)
        r["t"] = float(t)
        rows.append(r)
    tab = pd.DataFrame(rows)
    log("stage2 threshold on proxy:\n" + tab.to_string())
    m, ev, tm = make_proxy(i1, ip, ti1, tip, n_tr, sa, sb)
    pi1, pip, pp = i1[m], ip[m], p2[m]
    o2o = one_to_one(pi1, pip, pp)
    k = expected_f_decode(pi1[o2o], pp[o2o], n_tr)
    res["stage2_expF_proxy"] = score_int(ev, pi1[o2o][k], pip[o2o][k], ti1[tm], tip[tm], n_tr)
    best = tab.loc[tab["f"].idxmax()]
    res["stage2_thr_proxy"] = best.to_dict()
    log(f"stage2 proxy: best thr {best['t']} F={best['f']:.4f}; expF F={res['stage2_expF_proxy']['f']:.4f}")
    json.dump(res, open(Path(cache) / "proxy_report.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
