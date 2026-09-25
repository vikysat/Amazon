"""Stage: load raw sources and add normalised fields; cached per source as parquet."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from io_utils import log, read_tsv  # noqa: E402
from normalize import normalize_frame  # noqa: E402


def prepared_path(cache_dir, split: str, src: str) -> Path:
    """Path of the cached normalised parquet for (split, source)."""
    return Path(cache_dir) / "norm" / f"{split}_{src}.parquet"


def prepare(data_dir, cache_dir, split: str, n_jobs: int = 12) -> None:
    """Normalise s1/s2/s3 of a split (skips sources already cached)."""
    (Path(cache_dir) / "norm").mkdir(parents=True, exist_ok=True)
    for src in ("s1", "s2", "s3"):
        out = prepared_path(cache_dir, split, src)
        if out.exists():
            continue
        raw_cache = Path(cache_dir) / "raw" / f"{split}_{src}.parquet"
        if raw_cache.exists():
            df = pd.read_parquet(raw_cache)
        else:
            df = read_tsv(Path(data_dir) / split / f"{split}_source{src[1]}.tsv")
        log(f"normalising {split}/{src} ({len(df):,})")
        df = normalize_frame(df, n_jobs=n_jobs)
        df.to_parquet(out, index=False)
        log(f"wrote {out}")
        del df


def load_prepared(cache_dir, split: str, src: str, columns=None) -> pd.DataFrame:
    """Read a normalised source parquet (optionally a subset of columns)."""
    return pd.read_parquet(prepared_path(cache_dir, split, src), columns=columns)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--split", default="both")
    a = ap.parse_args()
    for sp in (["train", "test"] if a.split == "both" else [a.split]):
        prepare(a.data_dir, a.cache_dir, sp)
