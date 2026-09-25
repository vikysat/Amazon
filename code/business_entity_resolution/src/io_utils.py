"""Data loading / caching helpers.

All raw TSVs are read exactly as the challenge mandates
(``sep="\\t", dtype=str, keep_default_na=False``) and cached as parquet with
pyarrow-backed strings so that repeated runs are fast and memory-lean.
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
COLS = ["entity_id", "business_name", "business_address", "country"]


def seed_everything(seed: int = SEED) -> None:
    """Seed python's and numpy's global RNGs for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def log(msg: str) -> None:
    """Print a timestamped log line including current process RSS (GB)."""
    try:
        import psutil

        rss = psutil.Process().memory_info().rss / 1e9
        mem = f" [rss={rss:.1f}GB]"
    except Exception:  # psutil optional
        mem = ""
    print(f"{time.strftime('%H:%M:%S')}{mem} {msg}", flush=True)


def read_tsv(path: str | Path, chunksize: int = 1_000_000) -> pd.DataFrame:
    """Read a challenge TSV with the mandated options, chunked to limit peak memory.

    Each chunk is converted to ``string[pyarrow]`` dtype which is ~3x smaller
    than python object strings.
    """
    parts = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                             quoting=csv.QUOTE_NONE, chunksize=chunksize):
        parts.append(chunk.astype("string[pyarrow]"))
    return pd.concat(parts, ignore_index=True)


def load_split(data_dir: str | Path, split: str, cache_dir: str | Path) -> dict:
    """Load the three sources (and ground truth for train) of one split.

    Returns a dict with keys ``s1``, ``s2``, ``s3`` (DataFrames) and, for the
    train split, ``gt`` (DataFrame with source1_entity_id / matched_entity_ids).
    Results are cached as parquet under ``cache_dir/raw``.
    """
    data_dir, cache_dir = Path(data_dir), Path(cache_dir) / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    names = {"s1": "source1", "s2": "source2", "s3": "source3"}
    if split == "train":
        names["gt"] = "ground_truth"
    for key, suffix in names.items():
        cp = cache_dir / f"{split}_{key}.parquet"
        if cp.exists():
            df = pd.read_parquet(cp)
            df = df.astype("string[pyarrow]")
        else:
            df = read_tsv(data_dir / split / f"{split}_{suffix}.tsv")
            df.to_parquet(cp, index=False)
        out[key] = df
        log(f"loaded {split}/{key}: {len(df):,} rows")
    return out


def gt_pairs(gt: pd.DataFrame) -> pd.DataFrame:
    """Explode the ground-truth table into a long (s1, cand) pair table."""
    g = gt[gt["matched_entity_ids"].str.len() > 0]
    s = g["matched_entity_ids"].astype(str).str.split(",")
    long = pd.DataFrame({"s1": g["source1_entity_id"].astype(str).values, "cand": s.values}).explode("cand")
    return long.reset_index(drop=True)


def write_id_lists(path: str | Path, s1_ids, mapping: dict, col: str) -> None:
    """Write a results-style TSV: one row per S1 id, comma-joined id list.

    ``mapping`` maps s1 id -> iterable of S2/S3 ids (order preserved, de-duplicated).
    Written with csv.QUOTE_NONE as required by the challenge.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar=None, lineterminator="\n")
        w.writerow(["source1_entity_id", col])
        for s1 in s1_ids:
            ids = mapping.get(s1, ())
            seen, lst = set(), []
            for x in ids:
                if x not in seen:
                    seen.add(x)
                    lst.append(x)
            w.writerow([s1, ",".join(lst)])
