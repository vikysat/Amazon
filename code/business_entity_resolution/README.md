# Business Entity Resolution — Amazon ML Challenge 2026

Blocking + two-stage LightGBM pipeline that matches every Source-1 business to its Source-2/3
records. No external data, APIs or look-ups are used; the only inputs are the provided TSVs.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate      # (Linux/macOS: source .venv/bin/activate)
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/run_pipeline.py \
    --data-dir student_resource/dataset --out-dir output --cache-dir cache
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

`--data-dir` must contain `train/` and `test/` with the original file names. `--mode baseline`
runs the Phase-4 safety-net (string-similarity average + global threshold) instead of the model.

## Pipeline (data → blocking → matching → output)

| stage | module | cached artefact |
|---|---|---|
| load TSV (`sep="\t", dtype=str, keep_default_na=False`) | `io_utils.py` | `cache/raw/*.parquet` |
| normalisation (Indic transliteration, legal/street/region canonicalisation, postcode, numbers, city) | `normalize.py`, `prepare.py` | `cache/norm/*.parquet` |
| blocking: bidirectional sparse TF-IDF top-K within country | `blocking.py` | `cache/blocks/*.parquet` |
| pairwise features (rapidfuzz, IDF, numbers, postcode …) | `features.py` | `cache/feats/*.parquet` |
| stage-1 + stage-2 (relational) LightGBM, 5-fold GroupKFold OOF | `model.py` | `cache/stage*.txt` |
| decoding: one-to-one + threshold / expected-F0.5 | `decode.py` | – |
| outputs (`csv.QUOTE_NONE`) | `run_pipeline.py` | `output/*.tsv` |

Metric implementation and self-test: `python src/metrics.py`. EDA: `python src/eda.py`.
Every experiment appends a row to `experiments.csv` at the repo root.

## Hardware / runtime
Intel i7-13620H (10C/16T), 16 GB RAM, no GPU, Windows 11, Python 3.11.
Approximate end-to-end runtime from scratch: see NOTES.md (filled in at the final build).

## Notes
- Seeds are fixed (`io_utils.SEED = 42`); LightGBM runs with `deterministic=True`.
- The cross-encoder phase was skipped (no GPU available).
