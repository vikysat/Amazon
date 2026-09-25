"""Single entrypoint: data -> normalisation -> blocking -> features -> model -> decoding -> outputs.

Usage:
    python src/run_pipeline.py --data-dir <dataset dir> --out-dir <output dir> [--cache-dir cache]
                               [--mode model|baseline]

Intermediate artefacts are cached as parquet in ``--cache-dir`` so reruns are fast.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from decode import one_to_one, score_int, tune_threshold  # noqa: E402
from explog import log_experiment  # noqa: E402
from io_utils import log, seed_everything, write_id_lists  # noqa: E402
from prepare import load_prepared, prepare  # noqa: E402
from stages import blocking_report, feats_path, label_pairs, stage_block, stage_features, stage_filter  # noqa: E402


def write_outputs(out_dir, cache_dir, pairs: pd.DataFrame, keep: np.ndarray) -> None:
    """Write matching_results.tsv (pairs[keep]) and candidate_pairs.tsv (all pairs)."""
    s1 = load_prepared(cache_dir, "test", "s1", ["entity_id"])["entity_id"].astype(str).to_numpy()
    s2 = load_prepared(cache_dir, "test", "s2", ["entity_id"])["entity_id"].astype(str)
    s3 = load_prepared(cache_dir, "test", "s3", ["entity_id"])["entity_id"].astype(str)
    pool = pd.concat([s2, s3], ignore_index=True).to_numpy()
    df = pd.DataFrame({"s1": s1[pairs["i1"].to_numpy()], "c": pool[pairs["ip"].to_numpy()],
                       "p": pairs["p"].to_numpy() if "p" in pairs else 0.0})
    df = df.sort_values(["s1", "p"], ascending=[True, False])
    cand = df.groupby("s1", sort=False)["c"].agg(list).to_dict()
    match = df[keep[df.index.to_numpy()]].groupby("s1", sort=False)["c"].agg(list).to_dict()
    out_dir = Path(out_dir)
    write_id_lists(out_dir / "candidate_pairs.tsv", s1, cand, "candidate_entity_ids")
    write_id_lists(out_dir / "matching_results.tsv", s1, match, "matched_entity_ids")
    log(f"wrote outputs to {out_dir}: {int(keep.sum()):,} matches, {len(pairs):,} candidates")


def run_baseline(args) -> None:
    """Phase 4 baseline: mean(name token-set, address token-set) + one-to-one + global threshold."""
    tr = stage_filter(stage_block(args.cache_dir, "train", n_jobs=args.n_jobs))
    y, true = label_pairs(args.cache_dir, tr)
    n_s1 = len(load_prepared(args.cache_dir, "train", "s1", ["entity_id"]))
    stage_features(args.cache_dir, "train", tr, n_jobs=args.n_jobs, lazy=True)
    Xtr = pd.read_parquet(feats_path(args.cache_dir, "train", tr), columns=["n_tset", "a_tset"])
    p = ((Xtr["n_tset"].to_numpy() + Xtr["a_tset"].to_numpy()) / 200.0).astype(np.float32)
    ti1, tip = true["i1"].to_numpy(), true["ip"].to_numpy()
    i1, ip = tr["i1"].to_numpy(), tr["ip"].to_numpy()
    rep = blocking_report(tr, y, true, n_s1, 10_320_219)
    log(f"blocking report: {json.dumps(rep)}")
    t, tab = tune_threshold(np.arange(n_s1), i1, ip, p, ti1, tip, n_s1)
    best = tab.loc[tab["f"].idxmax()]
    log(f"baseline train F0.5={best.f:.4f} at t={t} (singleton acc {best.singleton_acc:.3f}, non-singleton {best.nonsingleton_f:.4f})")
    # cross-country: threshold tuned on one country, evaluated on the other
    c = load_prepared(args.cache_dir, "train", "s1", ["country"])["country"].astype(str).to_numpy()
    cc = []
    for a, b in (("US", "India"), ("India", "US")):
        ta, _ = tune_threshold(np.flatnonzero(c == a), i1, ip, p, ti1, tip, n_s1)
        m = one_to_one(i1, ip, p) & (p >= ta)
        cc.append(score_int(np.flatnonzero(c == b), i1[m], ip[m], ti1, tip, n_s1)["f"])
    log(f"cross-country F0.5 (US->India, India->US) = {cc}")
    log_experiment(Path(args.repo_root) / "experiments.csv", "phase4 baseline mean(n_tset,a_tset) o2o+threshold",
                   blocking_recall=rep["pair_recall"], cands_per_s1=rep["cands_per_s1"], oof_f05=float(best.f),
                   singleton_acc=float(best.singleton_acc), nonsingleton_f05=float(best.nonsingleton_f),
                   cross_country_f05=float(np.mean(cc)))
    # test
    te = stage_filter(stage_block(args.cache_dir, "test", n_jobs=args.n_jobs))
    stage_features(args.cache_dir, "test", te, n_jobs=args.n_jobs, lazy=True)
    Xte = pd.read_parquet(feats_path(args.cache_dir, "test", te), columns=["n_tset", "a_tset"])
    pt = ((Xte["n_tset"].to_numpy() + Xte["a_tset"].to_numpy()) / 200.0).astype(np.float32)
    te = te.assign(p=pt)
    keep = one_to_one(te["i1"].to_numpy(), te["ip"].to_numpy(), pt) & (pt >= t)
    write_outputs(args.out_dir, args.cache_dir, te, keep)


def main() -> None:
    """Parse arguments and run the requested pipeline mode."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--mode", default="model", choices=["model", "baseline"])
    ap.add_argument("--n-jobs", type=int, default=10)
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))
    args = ap.parse_args()
    seed_everything()
    for sp in ("train", "test"):
        prepare(args.data_dir, args.cache_dir, sp, n_jobs=args.n_jobs)
    if args.mode == "baseline":
        run_baseline(args)
    else:
        from model import run_model  # noqa: WPS433 (lazy import keeps baseline light)

        run_model(args)


if __name__ == "__main__":
    main()
