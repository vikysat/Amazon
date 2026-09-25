"""Stage-1 pairwise LightGBM + stage-2 relational LightGBM, OOF validation and test inference.

Validation: 5-fold GroupKFold over S1 entities (fold id = seeded random permutation of S1
rows). ALL S2/S3 records stay in the candidate pool in every fold (blocking is done once on the
full train set), so distractors look like test; only held-out S1 entities are evaluated.
Stage-2 relational features are always built from *out-of-fold* stage-1 probabilities.

Memory: one float32 matrix holds the pairwise features plus reserved relational columns. During
stage 1 the relational columns are NaN (constant -> never used by LightGBM); for stage 2 they are
filled with features derived from OOF stage-1 probabilities.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from decode import expected_f_decode, one_to_one, score_int, tune_threshold
from explog import log_experiment
from io_utils import SEED, log
from prepare import load_prepared
from stages import blocking_report, feats_path, label_pairs, stage_block, stage_features, stage_filter

N_FOLDS = 5
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
              max_bin=127, verbose=-1, seed=SEED, num_threads=12, deterministic=True,
              force_row_wise=True)
MAX_ROUNDS = 800
REL_COLS = ["c_rank", "c_n", "c_is_best", "c_margin", "c_other_best", "s_rank", "s_max", "s_gap",
            "s_n05", "s_sum", "s_n", "s_sum_best", "p1"]


def fold_of_s1(n_s1: int, seed: int = SEED) -> np.ndarray:
    """Fold id (0..N_FOLDS-1) for every S1 row, via a seeded random permutation."""
    rng = np.random.default_rng(seed)
    f = np.empty(n_s1, dtype=np.int8)
    f[rng.permutation(n_s1)] = np.arange(n_s1) % N_FOLDS
    return f


def relational_features(i1: np.ndarray, ip: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Stage-2 features from stage-1 probabilities ``p`` (must be OOF for train pairs).

    Per candidate record c (ip): rank of this S1 among c's S1s, is-best flag, margin to c's
    best other S1, the best other S1's prob, number of S1s proposing c.
    Per S1 entity s (i1): rank of c among s's candidates, max prob, gap to max, number of
    candidates above 0.5, prob sum, candidate count, and prob mass of candidates for which s is best.
    Column order equals REL_COLS.
    """
    df = pd.DataFrame({"i1": i1, "ip": ip, "p": p})
    g_ip = df.groupby("ip")["p"]
    df["c_rank"] = g_ip.rank(ascending=False, method="first").astype(np.float32)
    df["c_n"] = g_ip.transform("size").astype(np.float32)
    c_max = g_ip.transform("max").to_numpy()
    # second-highest prob per candidate record
    order = np.lexsort((-df["p"].to_numpy(), df["ip"].to_numpy()))
    ps, ips = df["p"].to_numpy()[order], df["ip"].to_numpy()[order]
    first = np.r_[True, ips[1:] != ips[:-1]]
    same_next = np.r_[ips[1:] == ips[:-1], False]
    sec_of_group = np.where(first & same_next, np.r_[ps[1:], 0.0], -1.0)[first]
    grp = np.cumsum(first) - 1
    second = np.empty(len(df), dtype=np.float32)
    second[order] = sec_of_group[grp]
    is_best = df["c_rank"].to_numpy() == 1
    other_best = np.where(is_best, second, c_max)
    df["c_is_best"] = is_best.astype(np.float32)
    df["c_margin"] = (df["p"].to_numpy() - np.maximum(other_best, 0)).astype(np.float32)
    df["c_other_best"] = other_best.astype(np.float32)
    g_s = df.groupby("i1")["p"]
    df["s_rank"] = g_s.rank(ascending=False, method="first").astype(np.float32)
    s_max = g_s.transform("max")
    df["s_max"] = s_max.astype(np.float32)
    df["s_gap"] = (s_max - df["p"]).astype(np.float32)
    df["s_n05"] = (df["p"] > 0.5).groupby(df["i1"]).transform("sum").astype(np.float32)
    df["s_sum"] = g_s.transform("sum").astype(np.float32)
    df["s_n"] = g_s.transform("size").astype(np.float32)
    df["s_sum_best"] = df["p"].where(is_best, 0.0).groupby(df["i1"]).transform("sum").astype(np.float32)
    df["p1"] = df["p"].astype(np.float32)
    return df[REL_COLS]


def load_matrix(path, n_rel: int = len(REL_COLS)) -> tuple[np.ndarray, list[str]]:
    """Load a feature parquet column-by-column into one float32 matrix with ``n_rel`` extra
    NaN columns reserved for stage-2 relational features (avoids a second big copy)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    X = np.empty((pf.metadata.num_rows, len(names) + n_rel), dtype=np.float32)
    for j, c in enumerate(names):
        X[:, j] = pf.read(columns=[c]).column(0).to_numpy()
    X[:, len(names):] = np.nan
    return X, names


def _dataset(X, y, names) -> lgb.Dataset:
    """Construct a LightGBM Dataset (binned once, reused via .subset for folds)."""
    return lgb.Dataset(X, label=y, feature_name=list(names), free_raw_data=False,
                       params={"max_bin": PARAMS["max_bin"], "verbose": -1}).construct()


def _predict(m, X, rows, chunk: int = 2_000_000) -> np.ndarray:
    """Predict the given rows of X in chunks (bounded memory)."""
    out = np.empty(len(rows), dtype=np.float32)
    for st in range(0, len(rows), chunk):
        out[st:st + chunk] = m.predict(X[rows[st:st + chunk]], num_threads=PARAMS["num_threads"])
    return out


def _train(ds_full: lgb.Dataset, tr_idx: np.ndarray, rng: np.random.Generator, rounds: int | None = None):
    """Train one LightGBM on the given rows; early stopping on a 10% inner split unless ``rounds``."""
    if rounds is not None:
        return lgb.train(PARAMS, ds_full.subset(np.sort(tr_idx)), num_boost_round=rounds)
    perm = rng.permutation(len(tr_idx))
    n_val = max(1, len(tr_idx) // 10)
    va, fit = np.sort(tr_idx[perm[:n_val]]), np.sort(tr_idx[perm[n_val:]])
    return lgb.train(PARAMS, ds_full.subset(fit), num_boost_round=MAX_ROUNDS,
                     valid_sets=[ds_full.subset(va)],
                     callbacks=[lgb.early_stopping(30, verbose=False)])


def cv_oof(X: np.ndarray, y: np.ndarray, groups_fold: np.ndarray, train_frac: float, i1: np.ndarray,
           name: str, feat_names, cache_dir=None) -> tuple[np.ndarray, list[int]]:
    """5-fold OOF predictions. Each fold trains on a ``train_frac`` sample of its training S1s.

    If ``cache_dir`` is given, OOF predictions and best iterations are cached there
    (``oof_<name>.npy`` / ``.json``) and reused on rerun.
    """
    if cache_dir is not None:
        fp, fj = Path(cache_dir) / f"oof_{name}.npy", Path(cache_dir) / f"oof_{name}.json"
        if fp.exists() and fj.exists():
            log(f"  {name}: using cached OOF {fp}")
            return np.load(fp), json.load(open(fj))
    rng = np.random.default_rng(SEED)
    ds = _dataset(X, y, feat_names)
    oof = np.zeros(len(y), dtype=np.float32)
    iters = []
    s1_keep = rng.random(i1.max() + 1) < train_frac
    for k in range(N_FOLDS):
        tr_idx = np.flatnonzero((groups_fold != k) & s1_keep[i1])
        te_idx = np.flatnonzero(groups_fold == k)
        b = _train(ds, tr_idx, rng)
        oof[te_idx] = _predict(b, X, te_idx)
        iters.append(b.best_iteration or MAX_ROUNDS)
        log(f"  {name} fold {k}: train rows {len(tr_idx):,}, best_iter {iters[-1]}")
    del ds
    if cache_dir is not None:
        np.save(fp, oof)
        json.dump(iters, open(fj, "w"))
    return oof, iters


def evaluate_decodings(i1, ip, p, ti1, tip, n_s1, eval_s1, missed=None) -> dict:
    """Tune and compare decodings on OOF probs; returns dict of results incl. best threshold."""
    res = {}
    t_raw, tab_raw = tune_threshold(eval_s1, i1, ip, p, ti1, tip, n_s1, o2o=False)
    res["thr"] = tab_raw.loc[tab_raw["f"].idxmax()].to_dict()
    t_o2o, tab = tune_threshold(eval_s1, i1, ip, p, ti1, tip, n_s1, o2o=True)
    res["o2o_thr"] = tab.loc[tab["f"].idxmax()].to_dict()
    o2o = one_to_one(i1, ip, p)
    m = expected_f_decode(i1[o2o], p[o2o], n_s1, missed_true=missed)
    res["o2o_expF"] = score_int(eval_s1, i1[o2o][m], ip[o2o][m], ti1, tip, n_s1)
    return res


def run_model(args) -> None:
    """Full model pipeline: CV on train (OOF metrics, cross-country), then fit-all and predict test."""
    from run_pipeline import write_outputs  # local import to avoid a cycle

    cache, repo = args.cache_dir, Path(args.repo_root)
    tr = stage_filter(stage_block(cache, "train", n_jobs=args.n_jobs))
    y, true = label_pairs(cache, tr)
    s1c = load_prepared(cache, "train", "s1", ["country"])["country"].astype(str).to_numpy()
    n_s1 = len(s1c)
    rep = blocking_report(tr, y, true, n_s1, 10_320_219)
    log(f"blocking report (after filter): {json.dumps(rep)}")
    stage_features(cache, "train", tr, n_jobs=args.n_jobs, lazy=True)
    X, f1 = load_matrix(feats_path(cache, "train", tr))
    names = f1 + REL_COLS
    nf = len(f1)
    log(f"train matrix {X.shape}")
    i1, ip = tr["i1"].to_numpy(), tr["ip"].to_numpy()
    ti1, tip = true["i1"].to_numpy(), true["ip"].to_numpy()
    fold = fold_of_s1(n_s1)[i1]
    frac = getattr(args, "train_frac", 0.3)
    rng = np.random.default_rng(SEED)
    keep_s1 = rng.random(n_s1) < frac
    all_s1 = np.arange(n_s1)

    # ---------------- stage 1 (relational columns are NaN -> unused)
    log("stage 1 CV")
    oof1, it1 = cv_oof(X, y, fold, frac, i1, "stage1", names, cache_dir=cache)
    d1 = evaluate_decodings(i1, ip, oof1, ti1, tip, n_s1, all_s1)
    log(f"stage1 OOF decodings: {json.dumps(d1, default=float)}")
    ds = _dataset(X, y, names)
    cc_p1, cc_rows = {}, {}
    for a, b in (("US", "India"), ("India", "US")):
        ta = np.flatnonzero((s1c[i1] == a) & keep_s1[i1])
        tb = np.flatnonzero(s1c[i1] == b)
        m = _train(ds, ta, rng, rounds=int(np.mean(it1)))
        cc_p1[b], cc_rows[b] = _predict(m, X, tb), tb
    keep_final = np.random.default_rng(SEED + 1).random(n_s1) < min(1.0, frac * 1.25)
    idx_all = np.flatnonzero(keep_final[i1])
    m1 = _train(ds, idx_all, rng, rounds=int(np.mean(it1) * 1.1))
    m1.save_model(str(Path(cache) / "stage1.txt"))
    del ds

    # ---------------- stage 2 (relational features from OOF stage-1 probs)
    log("stage 2 CV")
    X[:, nf:] = relational_features(i1, ip, oof1).to_numpy(np.float32)
    oof2, it2 = cv_oof(X, y, fold, frac, i1, "stage2", names, cache_dir=cache)
    d2 = evaluate_decodings(i1, ip, oof2, ti1, tip, n_s1, all_s1)
    log(f"stage2 OOF decodings: {json.dumps(d2, default=float)}")
    best_name = max(("o2o_thr", "o2o_expF"), key=lambda k: d2[k]["f"])
    best = d2[best_name]
    np.save(Path(cache) / "oof1.npy", oof1)
    np.save(Path(cache) / "oof2.npy", oof2)
    ds = _dataset(X, y, names)
    cc = {}
    for a, b in (("US", "India"), ("India", "US")):
        ta = np.flatnonzero((s1c[i1] == a) & keep_s1[i1])
        tb = cc_rows[b]
        m = _train(ds, ta, rng, rounds=int(np.mean(it2)))
        Xb = X[tb].copy()
        Xb[:, nf:] = relational_features(i1[tb], ip[tb], cc_p1[b]).to_numpy(np.float32)
        p2b = m.predict(Xb, num_threads=PARAMS["num_threads"])
        del Xb
        t_a, _ = tune_threshold(np.flatnonzero(s1c == a), i1, ip, oof2, ti1, tip, n_s1)
        msk = one_to_one(i1[tb], ip[tb], p2b) & (p2b >= t_a)
        cc[f"{a}->{b}"] = score_int(np.flatnonzero(s1c == b), i1[tb][msk], ip[tb][msk], ti1, tip, n_s1)["f"]
    log(f"cross-country F0.5: {cc}")
    m2 = _train(ds, idx_all, rng, rounds=int(np.mean(it2) * 1.1))
    m2.save_model(str(Path(cache) / "stage2.txt"))
    del ds, X
    log_experiment(repo / "experiments.csv", f"stage1+2 LGBM, decode={best_name}, frac={frac}",
                   blocking_recall=rep["pair_recall"], cands_per_s1=rep["cands_per_s1"], oof_f05=float(best["f"]),
                   singleton_acc=float(best["singleton_acc"]), nonsingleton_f05=float(best["nonsingleton_f"]),
                   cross_country_f05=float(np.mean(list(cc.values()))))
    json.dump({"stage1": d1, "stage2": d2, "cc": cc, "blocking": rep, "best": best_name},
              open(Path(cache) / "cv_report.json", "w"), default=float, indent=1)

    # ---------------- test inference
    te = stage_filter(stage_block(cache, "test", n_jobs=args.n_jobs))
    stage_features(cache, "test", te, n_jobs=args.n_jobs, lazy=True)
    Xt, _ = load_matrix(feats_path(cache, "test", te))
    ei1, eip = te["i1"].to_numpy(), te["ip"].to_numpy()
    rows = np.arange(len(te))
    p1 = _predict(m1, Xt, rows)
    Xt[:, nf:] = relational_features(ei1, eip, p1).to_numpy(np.float32)
    p2 = _predict(m2, Xt, rows)
    del Xt
    n_te = len(load_prepared(cache, "test", "s1", ["entity_id"]))
    o2o = one_to_one(ei1, eip, p2)
    if best_name == "o2o_thr":
        keep = o2o & (p2 >= best["t"])
    else:
        keep = np.zeros(len(p2), bool)
        sub = np.flatnonzero(o2o)
        keep[sub[expected_f_decode(ei1[sub], p2[sub], n_te)]] = True
    te = te.assign(p=p2)
    te[["i1", "ip", "p"]].to_parquet(Path(cache) / "test_scored.parquet", index=False)
    write_outputs(args.out_dir, cache, te, keep)
    # per-country predicted match-count distribution (Phase 9 sanity check)
    tc = load_prepared(cache, "test", "s1", ["country"])["country"].astype(str).to_numpy()
    cnt = np.bincount(ei1[keep], minlength=n_te)
    for c in np.unique(tc):
        vc = pd.Series(np.minimum(cnt[tc == c], 8)).value_counts(normalize=True).sort_index().round(3)
        log(f"test predicted match counts [{c}]: {vc.to_dict()}")
