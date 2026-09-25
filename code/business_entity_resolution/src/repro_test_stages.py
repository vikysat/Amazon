"""Reproducibility check for the slow stages (normalisation -> test blocking -> test features).

Runs the stages from scratch into ``--cache-dir`` and prints content hashes of the resulting
parquet tables, to be compared with the hashes produced on another machine.

Usage: python src/repro_test_stages.py --data-dir <dataset> --cache-dir cache_repro [--n-jobs 12]
       python src/repro_test_stages.py --cache-dir cache --hash-only   # hash an existing cache
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from io_utils import log, seed_everything  # noqa: E402
from prepare import prepare  # noqa: E402
from stages import feats_path, stage_block, stage_features, stage_filter  # noqa: E402


def table_hash(df: pd.DataFrame) -> str:
    """Order-sensitive SHA-256 over all columns (numeric columns rounded to 1e-5)."""
    h = hashlib.sha256()
    for c in df.columns:
        v = df[c].to_numpy()
        if np.issubdtype(v.dtype, np.floating):
            v = np.round(v.astype(np.float64), 5)
        h.update(c.encode())
        h.update(np.ascontiguousarray(v).tobytes() if v.dtype != object else "\x00".join(map(str, v)).encode())
    return h.hexdigest()[:16]


def main() -> None:
    """Run (or just hash) the test normalisation/blocking/feature stages."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir")
    ap.add_argument("--cache-dir", default="cache_repro")
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--hash-only", action="store_true")
    a = ap.parse_args()
    seed_everything()
    if not a.hash_only:
        prepare(a.data_dir, a.cache_dir, "test", n_jobs=a.n_jobs)
    blocks = stage_block(a.cache_dir, "test", n_jobs=a.n_jobs)
    log(f"HASH blocks/test_v1: {table_hash(blocks)}  rows={len(blocks):,}")
    te = stage_filter(blocks)
    if not a.hash_only:
        stage_features(a.cache_dir, "test", te, n_jobs=a.n_jobs, lazy=True)
    feats = pd.read_parquet(feats_path(a.cache_dir, "test", te))
    log(f"HASH feats/test: {table_hash(feats)}  shape={feats.shape}")
    norm = pd.read_parquet(Path(a.cache_dir) / "norm" / "test_s1.parquet")
    log(f"HASH norm/test_s1: {table_hash(norm)}  rows={len(norm):,}")


if __name__ == "__main__":
    main()
