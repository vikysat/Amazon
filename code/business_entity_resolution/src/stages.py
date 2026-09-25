"""Cached pipeline stages shared by experiments and the final run.

Each stage writes a parquet under ``cache_dir`` and is skipped when that file exists.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from blocking import tfidf_block
from features import compute_features, name_idf
from io_utils import gt_pairs, log
from prepare import load_prepared, prepare

BLOCK_PARAMS = dict(k_pool=3, k_s1=10, df_cap=1000, char_n=4)


def load_tables(cache_dir, split: str, columns=None):
    """Return (s1, pool) normalised tables; pool = S2 rows then S3 rows."""
    s1 = load_prepared(cache_dir, split, "s1", columns)
    s2 = load_prepared(cache_dir, split, "s2", columns)
    s3 = load_prepared(cache_dir, split, "s3", columns)
    pool = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    return s1, pool


def stage_block(cache_dir, split: str, n_jobs: int = 8, tag: str = "v1", params=None) -> pd.DataFrame:
    """Blocking stage (cached): returns candidate pairs (i1, ip, cos, r_p, r_s)."""
    params = params or BLOCK_PARAMS
    out = Path(cache_dir) / "blocks" / f"{split}_{tag}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    s1, pool = load_tables(cache_dir, split, ["entity_id", "country", "name_skel", "name_core", "addr_no_landmark"])
    log(f"blocking {split}: s1={len(s1):,} pool={len(pool):,} params={params}")
    pairs = tfidf_block(s1, pool, n_jobs=n_jobs, **params)
    pairs.to_parquet(out, index=False)
    log(f"blocking {split} done: {len(pairs):,} pairs ({len(pairs) / len(s1):.1f}/S1)")
    return pairs


def label_pairs(cache_dir, pairs: pd.DataFrame, data_dir=None) -> tuple[np.ndarray, pd.DataFrame]:
    """Attach 0/1 labels to train pairs; also return the true pair table as (i1, ip)."""
    s1 = load_prepared(cache_dir, "train", "s1", ["entity_id"])
    s2 = load_prepared(cache_dir, "train", "s2", ["entity_id"])
    s3 = load_prepared(cache_dir, "train", "s3", ["entity_id"])
    pool_ids = pd.concat([s2["entity_id"], s3["entity_id"]], ignore_index=True).astype(str)
    gt = pd.read_parquet(Path(cache_dir) / "raw" / "train_gt.parquet")
    tp = gt_pairs(gt)
    m1 = pd.Series(np.arange(len(s1)), index=s1["entity_id"].astype(str).values)
    mp = pd.Series(np.arange(len(pool_ids)), index=pool_ids.values)
    true = pd.DataFrame({"i1": m1.reindex(tp["s1"].values).to_numpy(),
                         "ip": mp.reindex(tp["cand"].values).to_numpy()}).astype(np.int64)
    key_true = true["i1"].to_numpy() * 20_000_000 + true["ip"].to_numpy()
    key_pairs = pairs["i1"].to_numpy().astype(np.int64) * 20_000_000 + pairs["ip"].to_numpy()
    y = np.isin(key_pairs, key_true).astype(np.int8)
    return y, true


def blocking_report(pairs: pd.DataFrame, y: np.ndarray, true: pd.DataFrame, n_s1: int, n_pool: int) -> dict:
    """Pair recall, per-entity recall ceiling (mean of per-entity recall), cands/S1, reduction ratio."""
    n_true = len(true)
    pair_recall = y.sum() / n_true
    per_true = true.groupby("i1").size()
    per_hit = pairs.loc[y == 1].groupby("i1").size().reindex(per_true.index, fill_value=0)
    ent_recall = float((per_hit / per_true).mean())
    rep = {
        "pairs": int(len(pairs)),
        "pair_recall": float(pair_recall),
        "entity_recall_mean": ent_recall,
        "entity_full_recall": float((per_hit == per_true).mean()),
        "cands_per_s1": len(pairs) / n_s1,
        "reduction_ratio": 1 - len(pairs) / (n_s1 * n_pool),
    }
    for col in ("r_p", "r_s"):
        if col in pairs:
            sel = pairs[col].to_numpy() >= 0
            rep[f"recall_{col}_only"] = float(y[sel].sum() / n_true)
    return rep


def feats_path(cache_dir, split: str, pairs: pd.DataFrame, tag: str = "v1") -> Path:
    """Cache path of the feature table; keyed by pair count so a changed filter invalidates it."""
    return Path(cache_dir) / "feats" / f"{split}_{tag}_{len(pairs)}.parquet"


def stage_features(cache_dir, split: str, pairs: pd.DataFrame, tag: str = "v1", n_jobs: int = 12,
                   lazy: bool = False) -> pd.DataFrame | None:
    """Pairwise feature stage (cached). With ``lazy`` only ensures the cache file exists."""
    out = feats_path(cache_dir, split, pairs, tag)
    if out.exists():
        return None if lazy else pd.read_parquet(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    s1, pool = load_tables(cache_dir, split)
    log(f"features {split}: {len(pairs):,} pairs")
    idf = name_idf(s1, pool)
    X = compute_features(pairs, s1, pool, idf, n_jobs=n_jobs)
    X.to_parquet(out, index=False)
    log(f"features {split} done")
    return None if lazy else X


# Final candidate filter applied after blocking (tuned on train recall vs. size).
# A pair is kept if ANY rule fires. The kept set is exactly what the model scores and what
# candidate_pairs.tsv contains.
FILTER = dict(rp_max=0, rs_max=4, cos_min=0.35)


def stage_filter(pairs: pd.DataFrame, rule: dict | None = None) -> pd.DataFrame:
    """Keep pairs that are pool->S1 rank <= rp_max, or S1->pool rank <= rs_max, or cos >= cos_min."""
    r = rule or FILTER
    rp, rs, cs = pairs["r_p"].to_numpy(), pairs["r_s"].to_numpy(), pairs["cos"].to_numpy()
    keep = ((rp >= 0) & (rp <= r["rp_max"])) | ((rs >= 0) & (rs <= r["rs_max"])) | (cs >= r["cos_min"])
    out = pairs.loc[keep].reset_index(drop=True)
    log(f"filter {r}: {len(pairs):,} -> {len(out):,} pairs")
    return out
