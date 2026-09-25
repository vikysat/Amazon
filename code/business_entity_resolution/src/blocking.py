"""Candidate generation (blocking).

Blocker "tfidf": sparse TF-IDF cosine over a token bag combining
  * name skeleton tokens (phonetic consonant skeletons, prefix ``n_``), and
  * address tokens (canonicalised, landmark-free, prefix ``a_``),
computed *within the same country string* (EDA: country agrees in 100% of true pairs;
this is generic string equality, so an unseen country like France just forms its own block).

Two directions are unioned, each tagged so the tags become model features:
  * pool->S1 (``r_p``): each S2/S3 record retrieves its top-``k_pool`` S1 entities. Because each
    S2/S3 record belongs to at most one S1 (EDA), this direction has very high recall per pair.
  * S1->pool (``r_s``): each S1 entity retrieves its top-``k_s1`` S2/S3 records.

Very frequent tokens (document frequency > ``df_cap``) are dropped from the sparse product
(they carry little IDF weight and dominate the cost).

Output: a DataFrame of candidate pairs with integer row indices into the S1 table and the
pool table (pool = S2 rows followed by S3 rows), plus cosine and rank features.
"""
from __future__ import annotations

import os
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from io_utils import log

_G = {}  # per-worker globals (index matrix loaded from memmap)


def block_doc(name_skel: pd.Series, addr: pd.Series, name_core: pd.Series | None = None,
              char_n: int = 0) -> list[str]:
    """Build the blocking document string for each record.

    Tokens: name-skeleton unigrams+bigrams (``n_``), address unigrams+bigrams (``a_``) and,
    if ``char_n`` > 0, character n-grams of the space-free name core (``c_``) which make
    the blocker robust to typos and concatenated names ("federalcenter.com").
    """
    ns = name_skel.fillna("").astype(str).str.split().tolist()
    ad = addr.fillna("").astype(str).str.split().tolist()
    nc = name_core.fillna("").astype(str).str.replace(" ", "", regex=False).tolist() if char_n else None
    out = []
    for j, (x, y) in enumerate(zip(ns, ad)):
        t = ["n_" + a for a in x] + ["n_" + a + "_" + b for a, b in zip(x, x[1:])]
        t += ["a_" + a for a in y] + ["a_" + a + "_" + b for a, b in zip(y, y[1:])]
        if char_n:
            w = nc[j]
            t += ["c_" + w[i:i + char_n] for i in range(max(0, len(w) - char_n + 1))]
        out.append(" ".join(t))
    return out


def _init_worker(path: str, shape) -> None:
    """Worker initializer: memory-map the CSC-transposed index matrix."""
    data = np.load(os.path.join(path, "data.npy"), mmap_mode="r")
    indices = np.load(os.path.join(path, "indices.npy"), mmap_mode="r")
    indptr = np.load(os.path.join(path, "indptr.npy"), mmap_mode="r")
    _G["idx"] = sp.csr_matrix((data, indices, indptr), shape=shape)


def _topk_rows(c: sp.csr_matrix, k: int):
    """Return (row, col, value, rank) arrays of the top-k entries per row of csr ``c``."""
    c.sum_duplicates()
    n_per = np.diff(c.indptr)
    rows = np.repeat(np.arange(c.shape[0], dtype=np.int32), n_per)
    order = np.lexsort((-c.data, rows))
    rows_s = rows[order]
    start = np.repeat(c.indptr[:-1], n_per)
    rank = np.arange(len(order)) - start
    keep = rank < k
    sel = order[keep]
    return rows_s[keep], c.indices[sel], c.data[sel].astype(np.float32), rank[keep].astype(np.int16)


def _query_chunk(args):
    """Worker: multiply a query chunk by the index and keep top-k per query row."""
    q, k, offset = args
    c = (q @ _G["idx"]).tocsr()
    r, col, v, rk = _topk_rows(c, k)
    return r + offset, col, v, rk


def sparse_topk(query: sp.csr_matrix, index: sp.csr_matrix, k: int, n_jobs: int = 8,
                chunk: int = 2000):
    """Top-k cosine neighbours of each query row among index rows (both L2-normalised).

    Runs chunked over ``n_jobs`` worker processes that share the index matrix via memmap.
    Returns (query_row, index_row, score, rank) arrays.
    """
    idx_t = index.T.tocsr().astype(np.float32)  # V x N_index
    tmp = tempfile.mkdtemp(prefix="blk_")
    np.save(os.path.join(tmp, "data.npy"), idx_t.data)
    np.save(os.path.join(tmp, "indices.npy"), idx_t.indices)
    np.save(os.path.join(tmp, "indptr.npy"), idx_t.indptr)
    shape = idx_t.shape
    del idx_t
    query = query.astype(np.float32).tocsr()
    tasks = [(query[i:i + chunk], k, i) for i in range(0, query.shape[0], chunk)]
    out = []
    with Pool(n_jobs, initializer=_init_worker, initargs=(tmp, shape)) as pool:
        for j, res in enumerate(pool.imap(_query_chunk, tasks, chunksize=4)):
            out.append(res)
            if j % 250 == 0:
                log(f"      topk chunk {j}/{len(tasks)}")
    for f in ("data.npy", "indices.npy", "indptr.npy"):
        try:
            os.remove(os.path.join(tmp, f))
        except OSError:
            pass
    return tuple(np.concatenate([o[i] for o in out]) for i in range(4))


def tfidf_block(s1: pd.DataFrame, pool: pd.DataFrame, k_pool: int = 3, k_s1: int = 10,
                df_cap: int = 1000, char_n: int = 0, n_jobs: int = 8) -> pd.DataFrame:
    """Run the bidirectional TF-IDF blocker within each country.

    ``s1`` / ``pool`` must contain ``country, name_skel, addr_no_landmark`` (+ ``name_core``
    when ``char_n`` > 0).
    Returns pairs (i1, ip, cos, r_p, r_s) with -1 rank where a direction did not propose it.
    """
    res = []
    for country in sorted(set(s1["country"].astype(str).unique())):
        i1 = np.flatnonzero((s1["country"] == country).to_numpy())
        ip = np.flatnonzero((pool["country"] == country).to_numpy())
        if len(i1) == 0 or len(ip) == 0:
            continue
        log(f"  block country={country}: s1={len(i1):,} pool={len(ip):,}")
        d1 = block_doc(s1["name_skel"].iloc[i1], s1["addr_no_landmark"].iloc[i1],
                       s1["name_core"].iloc[i1] if char_n else None, char_n)
        dp = block_doc(pool["name_skel"].iloc[ip], pool["addr_no_landmark"].iloc[ip],
                       pool["name_core"].iloc[ip] if char_n else None, char_n)
        vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True,
                              dtype=np.float32, max_df=df_cap / max(len(d1), 1) if len(d1) > df_cap else 1.0,
                              norm="l2")
        x1 = vec.fit_transform(d1)
        xp = vec.transform(dp)
        del d1, dp
        log(f"    vocab={len(vec.vocabulary_):,} nnz1={x1.nnz:,} nnzp={xp.nnz:,}")
        qr, ir, sc, rk = sparse_topk(xp, x1, k_pool, n_jobs=n_jobs)
        a = pd.DataFrame({"i1": i1[ir], "ip": ip[qr], "cos_p": sc, "r_p": rk})
        log(f"    pool->s1 pairs={len(a):,}")
        if k_s1 > 0:
            qr, ir, sc, rk = sparse_topk(x1, xp, k_s1, n_jobs=n_jobs)
            b = pd.DataFrame({"i1": i1[qr], "ip": ip[ir], "cos_s": sc, "r_s": rk})
            log(f"    s1->pool pairs={len(b):,}")
            m = a.merge(b, on=["i1", "ip"], how="outer")
        else:  # pool->S1 only (cheaper; nearly the same recall at equal candidate budget)
            b = None
            m = a.assign(cos_s=np.nan, r_s=-1)
        m["cos"] = m["cos_p"].fillna(m["cos_s"]).astype(np.float32)
        m["r_p"] = m["r_p"].fillna(-1).astype(np.int16)
        m["r_s"] = m["r_s"].fillna(-1).astype(np.int16)
        res.append(m[["i1", "ip", "cos", "r_p", "r_s"]])
        del x1, xp, a, b
    out = pd.concat(res, ignore_index=True)
    out["i1"] = out["i1"].astype(np.int32)
    out["ip"] = out["ip"].astype(np.int32)
    return out
