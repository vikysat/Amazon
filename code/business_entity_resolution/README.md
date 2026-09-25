# Business Entity Resolution — Amazon ML Challenge 2026

Blocking + two-stage LightGBM pipeline that matches every Source-1 business to its Source-2/3
records. No external data, APIs or look-ups are used; the only inputs are the provided TSVs.
Out-of-fold macro F0.5 on train: **0.9735** (details in `Documentation_template.md` and `NOTES.md`).

## Reproduce (data → blocking → matching → output)

From the repository root, with the challenge data at `student_resource/dataset/{train,test}/`:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate                     # Linux/macOS: source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/run_pipeline.py \
    --data-dir student_resource/dataset --out-dir output --cache-dir cache --n-jobs 12
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test        # must print PASS
```

Options: `--mode baseline` runs the Phase-4 safety net (string-similarity average + one-to-one +
global threshold). `--n-jobs` sets worker processes (lower it on machines with < 16 GB RAM).
Every stage caches its result as parquet under `--cache-dir`; delete a file to recompute that stage.

| stage | module | cached artefact | time* |
|---|---|---|---|
| read TSV (`sep="\t", dtype=str, keep_default_na=False`) | `io_utils.py` | `cache/raw/*.parquet` | 3 min |
| normalisation | `normalize.py`, `prepare.py` | `cache/norm/*.parquet` | 25 min |
| blocking (pool→S1 sparse TF-IDF top-6, within country) | `blocking.py`, `stages.py` | `cache/blocks/{train,test}_v1.parquet` | ~2 h + ~1.7 h |
| candidate filter (rank ≤ 1 or cos ≥ 0.4) | `stages.py` | – | 1 min |
| pairwise features | `features.py` | `cache/feats/*.parquet` | ~80 + 40 min |
| stage-1 + stage-2 LightGBM, 5-fold OOF, cross-country, final fit | `model.py` | `cache/stage{1,2}.txt`, `cache/oof_*.npy`, `cache/cv_report.json` | ~3 h |
| decoding (one-to-one + expected-F0.5) and outputs | `decode.py`, `run_pipeline.py` | `output/*.tsv`, `cache/test_scored.parquet` | 10 min |

\*Intel i7-13620H (10C/16T), 16 GB RAM, no GPU, Windows 11, Python 3.11. Total from scratch ≈ 7 h.
Peak RAM ≈ 10 GB (model stage). On Windows, raise the Python processes to AboveNormal priority or use
the "Best performance" power mode — background processes are otherwise throttled (EcoQoS).

Other entry points: `python src/metrics.py` (metric self-test: worked example = 0.714),
`python src/eda.py --data-dir ...` (Phase-0 EDA), `python src/run_proxy.py` (train/test composition
comparison), `python src/repro_test_stages.py` (hashes of the slow stages for reproducibility checks).
Every experiment appends a row to `experiments.csv` at the repo root.

## Notes
- Seeds are fixed (`io_utils.SEED = 42`); LightGBM runs with `deterministic=True`.
- Country is treated as an open set of strings: it only partitions blocking (true pairs always share
  it); no code path depends on the country values seen in train.
- Phase 7 (neural cross-encoder) was skipped: no GPU available. No pretrained neural models are used.
- Models / licenses: LightGBM (MIT); scikit-learn (BSD-3), rapidfuzz (MIT), pandas/numpy/scipy (BSD),
  pyarrow (Apache-2.0).
