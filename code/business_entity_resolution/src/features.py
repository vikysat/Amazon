"""Pairwise features for (S1, candidate) pairs.

Computed chunk-wise in worker processes with rapidfuzz. Strings are gathered in the parent
per chunk so that memory stays bounded. Every feature is country-agnostic (the only
country-derived signal would be an equality flag, which is constant 1 because blocking is
within-country, so it is omitted).
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from io_utils import log

STR_COLS = ["name_core", "name_skel", "name_legal", "name_norm", "addr_norm", "addr_no_landmark",
            "addr_numbers", "postcode", "city", "business_name"]

_IDF = {}


def _init(idf: dict) -> None:
    """Worker initializer: install the name-token IDF table."""
    _IDF.clear()
    _IDF.update(idf)


def _acronym(a_toks: list[str], b_toks: list[str]) -> int:
    """1 if the initials of one name equal a token of the other (len>=2)."""
    if len(a_toks) >= 2:
        ia = "".join(t[0] for t in a_toks)
        if ia in b_toks:
            return 1
    if len(b_toks) >= 2:
        ib = "".join(t[0] for t in b_toks)
        if ib in a_toks:
            return 1
    return 0


def _pair_feats(a: dict, b: dict, n: int) -> np.ndarray:
    """Compute the feature matrix for ``n`` pairs given column->list dicts for both sides."""
    idf = _IDF
    default_idf = idf.get("__max__", 10.0)
    out = np.zeros((n, len(FEATURES)), dtype=np.float32)
    for i in range(n):
        n1, n2 = a["name_core"][i], b["name_core"][i]
        k1, k2 = a["name_skel"][i], b["name_skel"][i]
        ad1, ad2 = a["addr_norm"][i], b["addr_norm"][i]
        al1, al2 = a["addr_no_landmark"][i], b["addr_no_landmark"][i]
        t1, t2 = n1.split(), n2.split()
        s1, s2 = set(t1), set(t2)
        sk1, sk2 = set(k1.split()), set(k2.split())
        inter = s1 & s2
        w_int = sum(idf.get(t, default_idf) for t in inter)
        w_uni = sum(idf.get(t, default_idf) for t in s1 | s2)
        w_1 = sum(idf.get(t, default_idf) for t in s1 - s2)
        w_2 = sum(idf.get(t, default_idf) for t in s2 - s1)
        sk_int = len(sk1 & sk2)
        l1, l2 = a["name_legal"][i], b["name_legal"][i]
        lg1, lg2 = set(l1.split()), set(l2.split())
        num1 = {t for t in t1 if t.isdigit()}
        num2 = {t for t in t2 if t.isdigit()}
        an1, an2 = a["addr_numbers"][i].split(), b["addr_numbers"][i].split()
        sa1, sa2 = set(an1), set(an2)
        p1, p2 = a["postcode"][i], b["postcode"][i]
        c1, c2 = a["city"][i], b["city"][i]
        at1, at2 = set(ad1.split()), set(ad2.split())
        at1w = {t for t in at1 if not t.isdigit()}
        at2w = {t for t in at2 if not t.isdigit()}
        out[i] = (
            fuzz.ratio(n1, n2), fuzz.partial_ratio(n1, n2), fuzz.token_sort_ratio(n1, n2),
            fuzz.token_set_ratio(n1, n2), JaroWinkler.similarity(n1, n2) * 100,
            fuzz.ratio(k1, k2), fuzz.token_set_ratio(k1, k2),
            fuzz.token_set_ratio(a["business_name"][i].lower(), b["business_name"][i].lower()),
            w_int / w_uni if w_uni > 0 else 0.0, w_int, w_1, w_2,
            len(inter), sk_int / max(1, len(sk1 | sk2)),
            _acronym(t1, t2),
            1.0 if (t1 and t2 and t1[0] == t2[0]) else 0.0,
            (1.0 if lg1 == lg2 else (-1.0 if (lg1 and lg2 and not (lg1 & lg2)) else 0.0)) if (lg1 or lg2) else 2.0,
            min(len(n1), len(n2)) / max(1, len(n1), len(n2)),
            (1.0 if num1 == num2 else -1.0) if (num1 or num2) else 0.0,
            fuzz.ratio(ad1, ad2), fuzz.token_set_ratio(ad1, ad2), fuzz.token_sort_ratio(ad1, ad2),
            fuzz.partial_ratio(ad1, ad2), fuzz.token_set_ratio(al1, al2),
            (1.0 if p1 == p2 else -1.0) if (p1 and p2) else 0.0,
            len(sa1 & sa2), len(sa1 & sa2) / max(1, len(sa1 | sa2)),
            (1.0 if an1[0] == an2[0] else -1.0) if (an1 and an2) else 0.0,
            len(sa1 - sa2), len(sa2 - sa1),
            fuzz.ratio(c1, c2) if (c1 and c2) else -1.0,
            len(at1w & at2w) / max(1, len(at1w | at2w)), len(at1w & at2w),
            len(ad1), len(ad2), len(n1), len(n2), len(t1), len(t2),
            1.0 if not ad2 else 0.0,
        )
    return out


FEATURES = [
    "n_ratio", "n_partial", "n_tsort", "n_tset", "n_jw",
    "sk_ratio", "sk_tset", "raw_tset",
    "n_idf_jac", "n_idf_int", "n_idf_only1", "n_idf_only2", "n_ntok_int", "sk_jac",
    "n_acronym", "n_first_tok", "legal_agree", "n_len_ratio", "n_num_agree",
    "a_ratio", "a_tset", "a_tsort", "a_partial", "anl_tset",
    "pc_agree", "num_int", "num_jac", "num_first_agree", "num_only1", "num_only2",
    "city_ratio", "street_jac", "street_int",
    "a_len1", "a_len2", "n_len1", "n_len2", "n_ntok1", "n_ntok2", "a_empty2",
]


def _worker(args):
    """Worker entry: unpack a chunk and compute features."""
    a, b, n = args
    return _pair_feats(a, b, n)


def name_idf(s1: pd.DataFrame, pool: pd.DataFrame) -> dict:
    """IDF of name_core tokens over S1 + pool (unsupervised; fine on test data)."""
    toks = pd.concat([s1["name_core"], pool["name_core"]], ignore_index=True).astype(str).str.split()
    ex = toks.explode()
    ex = ex[ex.notna() & (ex != "")]
    vc = ex.value_counts()
    n_docs = len(toks)
    idf = np.log((n_docs + 1) / (vc.to_numpy() + 1)) + 1
    d = dict(zip(vc.index.astype(str), idf.astype(float)))
    d["__max__"] = float(np.log(n_docs + 1) + 1)
    return d


def compute_features(pairs: pd.DataFrame, s1: pd.DataFrame, pool: pd.DataFrame, idf: dict,
                     n_jobs: int = 12, chunk: int = 20000) -> pd.DataFrame:
    """Compute FEATURES for every row of ``pairs`` (columns i1, ip) and return a DataFrame.

    Blocker-derived columns (cos, r_p, r_s) and the source flag are appended as-is.
    """
    c1 = {c: s1[c].astype(str).to_numpy(dtype=object) for c in STR_COLS}
    cp = {c: pool[c].astype(str).to_numpy(dtype=object) for c in STR_COLS}
    i1 = pairs["i1"].to_numpy()
    ip = pairs["ip"].to_numpy()

    def gen():
        """Yield per-chunk string dicts for the workers."""
        for st in range(0, len(pairs), chunk):
            a_idx, b_idx = i1[st:st + chunk], ip[st:st + chunk]
            yield ({c: c1[c][a_idx].tolist() for c in STR_COLS},
                   {c: cp[c][b_idx].tolist() for c in STR_COLS}, len(a_idx))

    parts = []
    with Pool(n_jobs, initializer=_init, initargs=(idf,)) as pool_:
        for k, r in enumerate(pool_.imap(_worker, gen(), chunksize=2)):
            parts.append(r)
            if k % 200 == 0:
                log(f"    features chunk {k} / {len(pairs) // chunk + 1}")
    X = pd.DataFrame(np.concatenate(parts), columns=FEATURES)
    for c in ("cos", "r_p", "r_s"):
        X[c] = pairs[c].to_numpy()
    X["is_s3"] = pool["entity_id"].astype(str).str.startswith("S3").to_numpy()[ip].astype(np.int8)
    return X
