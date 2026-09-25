# Teammate tasks (parallel work)

Read `CLAUDE.md` (task spec) and `NOTES.md` (current state) first. The main machine owns the
model/decoding work; **do not edit** `model.py`, `decode.py`, `features.py`, `stages.py`,
`blocking.py`. Work on a branch (`teammate`) and open a PR / send diffs.

## 0. Setup (~30 min)
```bash
git clone https://github.com/vikysat/Amazon && cd Amazon
git checkout -b teammate
# copy the dataset so that student_resource/dataset/{train,test}/*.tsv exist (not in git)
py -3.11 -m venv .venv            # Python 3.11
.venv\Scripts\activate
pip install -r code/business_entity_resolution/requirements.txt
```
Keep the laptop plugged in, power mode = Best performance (Windows throttles background
processes otherwise).

## Task A — reproducibility run of the slow stages (start FIRST; ~2.5-3.5 h, runs in background)
```bash
python -u code/business_entity_resolution/src/repro_test_stages.py --data-dir student_resource/dataset --cache-dir cache_repro --n-jobs 12 > repro.log 2>&1
```
When done, send back the three `HASH ...` lines at the end of `repro.log` (and the whole log).
Expected hashes from the main machine are in `REPRO_HASHES.txt`. Do not change code before this
run finishes (it must use the pushed code as-is).

## Task B — France normalisation review (Phase 9; ~1-1.5 h, while Task A runs)
- France appears only in test (259k S1, 1.43M S2/S3). Load `student_resource/dataset/test/*`,
  take ~50 French records per source and run them through
  `code/business_entity_resolution/src/normalize.py` (`normalize_name`, `normalize_address`).
- Look for: legal forms / noise words not stripped ("(France)", "Fils", "Frères", "Groupe", ...),
  street types not canonicalised ("R.", "Bd", "Anenue" typos, "Imp", "Pl"), départements vs regions
  (Gironde, Nord, Loire-Atlantique -> region codes), accents, "d'"/"l'" elisions, "bis/ter" numbers.
- Also compare what a French S1 looks like vs. its plausible S2/S3 counterparts (same street + number)
  to see which noise is injected.
- Deliverable: a diff to `normalize.py` that ONLY adds French-specific dictionary entries/regexes
  (must not change US/India outputs), plus a short findings list appended to a new file
  `notes_france.md`. Only hand-written general-knowledge dictionaries are allowed (no external data).

## Task C — documentation draft (Phase 10; ~1-1.5 h)
- Fill `Documentation_template.md` (repo root) from `NOTES.md`: methodology, EDA insights, blocking
  (recall / reduction ratio numbers are in NOTES.md), features (see `features.py` FEATURES list),
  model (stage-1 + stage-2 LightGBM, relational features), decoding (one-to-one + threshold /
  expected-F), validation (5-fold GroupKFold over S1, cross-country US<->India), models & licenses
  (LightGBM MIT, scikit-learn BSD, rapidfuzz MIT; no pretrained neural models), and an explicit
  statement that no external data or lookups were used. Leave `[TBD]` for final scores.
- Polish `code/business_entity_resolution/README.md` (exact commands, hardware, runtimes).
