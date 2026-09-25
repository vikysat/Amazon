"""Phase 0 exploratory data analysis. Prints the statistics recorded in NOTES.md.

Usage: python src/eda.py --data-dir <dataset> --cache-dir <cache>
"""
from __future__ import annotations

import argparse
import functools
print = functools.partial(print, flush=True)  # noqa: A001
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent))
from io_utils import gt_pairs, load_split, log, seed_everything  # noqa: E402


def main() -> None:
    """Run all EDA checks and print them."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", default="cache")
    a = ap.parse_args()
    seed_everything()
    tr = load_split(a.data_dir, "train", a.cache_dir)
    te = load_split(a.data_dir, "test", a.cache_dir)

    print("\n== Row counts per file x country ==")
    for split, d in (("train", tr), ("test", te)):
        for k in ("s1", "s2", "s3"):
            vc = d[k]["country"].value_counts()
            print(split, k, len(d[k]), dict(vc))

    gt = tr["gt"]
    n_match = gt["matched_entity_ids"].str.count(",").add(1).where(gt["matched_entity_ids"].str.len() > 0, 0)
    print("\n== Singleton rate:", float((n_match == 0).mean()))
    print("match count distribution:\n", n_match.value_counts().sort_index().head(20))

    pairs = gt_pairs(gt)
    pairs["src"] = pairs["cand"].str[:2]
    print("\n== S2 vs S3 share of matched pairs:", dict(pairs["src"].value_counts(normalize=True)))
    ct = pd.crosstab(pairs["s1"], pairs["src"])
    print("per-S1 (S2 count, S3 count) distribution:", ct.value_counts().head(15))

    dup = pairs["cand"].value_counts()
    print("\n== S2/S3 IDs under >1 S1:", int((dup > 1).sum()), "of", len(dup))

    all23 = pd.concat([tr["s2"], tr["s3"]], ignore_index=True)
    print("S2/S3 records never matched:", len(all23) - len(dup), "of", len(all23))
    s1 = tr["s1"].set_index("entity_id")
    o = all23.set_index("entity_id")
    pairs = pairs.join(s1, on="s1").join(o, on="cand", rsuffix="_c")
    print("\n== country agreement among matched pairs:", float((pairs["country"] == pairs["country_c"]).mean()))
    print(pd.crosstab(pairs["country"], pairs["country_c"]))

    samp = pairs.sample(30000, random_state=0).copy()
    samp["nsim"] = [fuzz.token_set_ratio(a_.lower(), b_.lower()) for a_, b_ in zip(samp["business_name"], samp["business_name_c"])]
    samp["asim"] = [fuzz.token_set_ratio(a_.lower(), b_.lower()) for a_, b_ in zip(samp["business_address"], samp["business_address_c"])]
    print("\nname token_set sim quantiles:", samp["nsim"].quantile([.01, .05, .1, .25, .5]).to_dict())
    print("addr token_set sim quantiles:", samp["asim"].quantile([.01, .05, .1, .25, .5]).to_dict())
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    cols = ["business_name", "business_name_c", "business_address", "business_address_c"]
    print("\n== 30 random matched pairs ==")
    for _, r in samp.head(30).iterrows():
        print(f"[{r.country}] {r.business_name!r:45} | {r.business_name_c!r}\n        {r.business_address!r} | {r.business_address_c!r}")
    print("\n== 10 hard pairs (low name sim) ==")
    for _, r in samp.nsmallest(10, "nsim").iterrows():
        print(f"[{r.country}] {r.business_name!r:45} | {r.business_name_c!r}\n        {r.business_address!r} | {r.business_address_c!r}")

    # empty fields
    for k in ("s1", "s2", "s3"):
        d = tr[k]
        print(k, "empty name", float((d["business_name"].str.len() == 0).mean()),
              "empty addr", float((d["business_address"].str.len() == 0).mean()))
    # non-latin share
    for k in ("s1", "s2", "s3"):
        d = tr[k]
        nl = d["business_name"].str.contains(r"[^\x00-ɏ]", regex=True)
        print(k, "non-latin name share", float(nl.mean()))
    fr = te["s1"][te["s1"]["country"] == "France"]
    print("\n== French S1 sample ==")
    print(fr.sample(20, random_state=1).to_string())


if __name__ == "__main__":
    main()
