# NOTES — Business Entity Resolution (Amazon ML Challenge 2026)

## Status (keep updated)
- **Current phase:** 5/6/8/9 done; Phase 10 docs written; reproducibility re-run (fresh clone) in progress
- **Best OOF F0.5:** 0.9735 (stage-2 LightGBM + one-to-one + expected-F) -> submissions/phase6_stage2_0.9735
  (validator PASS). Leaderboard: baseline 0.645; model [pending user submission].
- **Next step:** get model LB score; then optional full rebuild with France fixes (notes_france.md) and
  more boosting rounds / larger training fraction.

## Environment
- Hardware: Intel i7-13620H (10 cores / 16 threads), 15.7 GB RAM (often only ~4-8 GB free), no GPU (Intel UHD). Windows 11.
- Python 3.11 venv at `.venv/` (repo root). Data lives at `student_resource/dataset/` (git-ignored).
- Phase 7 (cross-encoder) is SKIPPED: no GPU (per instructions).

## Models used (name — license)
- LightGBM 4.7.0 — MIT (planned)
- scikit-learn — BSD-3, rapidfuzz — MIT
- No pretrained neural models used so far. (multilingual-e5-small, MIT, was considered for dense
  blocking but ~12M records x CPU embedding ≈ several hours; deferred.)

## Phase 0 — EDA (src/eda.py, output in cache_eda.txt)
Row counts:
| file | total | US | India | France |
|---|---|---|---|---|
| train s1 | 2,206,821 | 1,323,633 | 883,188 | – |
| train s2 | 5,034,616 | 3,016,817 | 2,017,799 | – |
| train s3 | 5,285,603 | 3,170,056 | 2,115,547 | – |
| test s1 | 1,732,544 | 663,106 | 809,986 | 259,452 |
| test s2 | 4,887,273 | 1,871,330 | 2,312,565 | 703,378 |
| test s3 | 5,082,316 | 1,945,701 | 2,405,000 | 731,615 |

- Singleton rate (train): **5.6%**. Match-count distribution: 0:123k, 1:119k, 2:375k, 3:531k, 4:484k,
  5:322k, 6:165k, 7:64k, 8:19k, 9:4k, 10+: 0.6k. Mean ≈ 3.5 matches per S1 → recall matters a lot.
- Matched pairs: S2 48.4%, S3 51.6%. Most entities have both S2 and S3 matches (1-3 each).
- **No S2/S3 id appears under more than one S1** (0 of 7.64M) → one-to-one constraint holds; exploit it
  (pool-side blocking + assignment of each S2/S3 to at most one S1).
- 2.68M of 10.32M S2/S3 records (26%) match no S1 → distractors.
- **Country agrees in 100% of matched pairs** → blocking within same country string (generic equality,
  works for unseen France).
- Name token_set similarity of true pairs: 1%:8, 5%:11, 10%:43, 25%:83, median 96. So ~5-10% of true
  pairs have a completely different name: (a) name in Indic script (Devanagari, Bengali, Telugu, Tamil,
  Kannada, Malayalam, Gujarati) — 9.4% of S2 names and 5.3% of S3 names are non-Latin; (b) name
  replaced by a random pseudo-word ("Nylaquo", "Quoquo", "Rizaaria") — then only the address matches.
- Address token_set similarity: 5%:52, 10%:70, median 91. 3.3% of S2/S3 addresses are empty.
- Noise patterns: US — typos ("Thbdr C0mpany"), word-order swaps, legal suffix add/drop/bracketed
  ("[LLC]", "L.L.C."), "F/K/A" trade names, domain-names as names ("holymethodistchurch.com"),
  state full name vs abbreviation, Ave/Avenue, unit/suite prefixes, number prefixes ("004669"),
  components reordered, city typos ("BELVIDEER", "WASHINGOTN"). India — upper-case variants, Pvt/Private,
  Ltd/Limited dropped, native-script names and states ("ಕರ್ನಾಟಕ"), address truncation, "Nr."/"Opp"
  landmarks, "null" fillers, house-number edits ("E-32" vs "E-34", "907" vs "07").
- France (test only): names like "Thermal & Fils SASU", "Association du Archives", "(France)" inserted;
  legal forms SARL/SAS/SASU/EURL/SCI/S.A.S; addresses "N Rue X, City, Region" with Rue/R./BD/Av
  abbreviations, typos ("Anenue"), region sometimes replaced by département (Gironde, Nord);
  no postcodes seen in the sample.

## Phase 1 — metric
- `src/metrics.py`: vectorised macro F0.5 on pair tables; self-test passes (worked example = 0.7143).

## Phase 2 — normalisation (src/normalize.py, src/prepare.py)
- NFKD + accent strip + lowercase, & / + -> "and", initials collapse ("l l c" -> llc, "s a s" -> sas).
- **Indic transliteration** (hand-written table): all Indic blocks are mapped to Devanagari by
  Unicode offset, then one Devanagari->Latin table with schwa handling (+ Malayalam chillus,
  Gurmukhi tippi). "डिजिटल इंफोटेक प्राइवेट लिमिटेड" -> "dijital inphotek pvt ltd".
- **Phonetic skeleton** of name tokens (name_skel): consonant skeleton with ph/f/p, v/w/b, g/j,
  c/k/q merges -> English and transliterated spellings coincide ("digital infotech" and
  "dijital inphotek" -> "djtl inptk"). Transliterated legal words detected by skeleton.
- Legal forms (US/IN/FR incl. SARL, SAS, SASU, SA, EURL, SCI, societe), street types (US/IN/FR incl.
  rue/r, bd/bld, av, chemin, allee, imp, rte), US states / Indian states / French regions canonicalised
  to short codes, ordinal words -> digits, leading zeros stripped from numbers, leetspeak undone in
  mixed alnum name tokens ("pr0jects", "5terling").
- Fields: name_norm, name_core, name_legal, name_skel, addr_norm, addr_no_landmark, addr_numbers,
  postcode (regex 5/6 digits, no country gating), city (best effort). Raw strings kept.
- Runtime: ~25 min for all 22M records with 12 processes.

## Phase 3 — blocking (src/blocking.py)
- Sparse TF-IDF (sublinear tf, L2) over a token bag: name-skeleton unigrams+bigrams, address
  unigrams+bigrams, char 4-grams of space-free name core; tokens with df > 1000 dropped from the
  product (speed; they carry little IDF weight). Always within the same country string.
- Word unigrams alone were ~10x slower and weaker; bigrams make the product sparse & fast.
- Sample study (20k S1 / their true pool records):
  - pool->S1 top-3: India 93.8% / US 97.4%; +S1->pool top-10: 94.9% / 98.2%.
  - pool->S1 top-5 alone: India 94.6% / US 97.9% at the same candidate budget -> S1->pool direction
    (4.7x bigger index, ~40 min/country) dropped; using pool->S1 top-6.
  - char 4-grams add +1.5-2pp recall (typos, concatenated names like "federalcenter.com").
- Residual misses: generic names + empty/truncated address (many S1s with the same name),
  gibberish replacement names with truncated addresses, script-switched names with address edits.

### Phase 3 full-scale result
- Train: pool->S1 top-6 = 61.9M pairs (28/S1), pair recall 96.8%. Test: 59.8M pairs (34.5/S1).
- Runtime: train ~2h20m, test ~1h45m (16 threads; Windows EcoQoS throttled background workers to
  ~20% until process priority was raised to AboveNormal).
- Final candidate filter (stages.FILTER): keep pool->S1 rank <= 1 OR cosine >= 0.4
  -> train 25.7M pairs (11.66/S1), **pair recall 96.0%**, per-entity recall mean 96.0%, 88.3% of
  entities have all true matches in candidates, reduction ratio 0.999999. Test: 26.2M pairs.
  (grid: rank0|cos>=.4: 8.5/S1 95.4%; rank<=2|cos>=.4: 15.4/S1 96.3%).

## Phase 4 — baseline (safety net)
- score = mean(token_set(name_core), token_set(addr_norm)) / 100, one-to-one assignment, global threshold.
- Train (no fitting except threshold) F0.5 = **0.7335** at t=0.80 (singleton acc 0.10, non-singleton 0.771);
  without one-to-one 0.7283. Cross-country (threshold from one country): US->India 0.7166, India->US 0.7308.
- Test output: 7.94M matches, 17.6k empty rows; validator PASS. Copied to submissions/phase4_baseline_0.7335/.
- Train features: 25.7M pairs x 44 features in ~80 min (10 procs, concurrent with test blocking).
- **Leaderboard: 0.645** (public) vs 0.7335 on train -> systematic shift, not noise. Test has
  ~5.75 pool records per S1 vs 4.68 in train (34.5 vs 28 raw candidates/S1) -> likely more
  singletons / unowned distractors; baseline has singleton acc 0.10, so it is hit hard. France unseen.
  -> building a test-like proxy validation (src/proxy_val.py) to calibrate decoding.

## Phase 5 — stage-1 LightGBM (src/model.py)
- 44 pairwise features + blocker cos/rank + source flag; LightGBM binary (lr 0.1, 127 leaves,
  min_data_in_leaf 200, feature/bagging fraction 0.8, lambda_l2 10, max_bin 127), early stopping
  on a 10% inner split, max 800 rounds (all folds hit ~800 -> more rounds could still help).
- 5-fold GroupKFold over S1; each fold trains on a 30% sample of its training S1 entities (~6.2M rows).
- **Stage-1 OOF F0.5 = 0.9691** (one-to-one + threshold 0.675; singleton acc 0.964, non-singleton
  0.969). Without one-to-one: 0.9686 (t=0.70). Expected-F decoding: 0.9686 (singleton acc 0.946).
- Runtime: ~6-12 min per fold on this laptop (paging: 25.7M x 57 float32 matrix = 5.9 GB).

## Phase 6 — stage-2 relational LightGBM
- Relational features from OOF stage-1 probs (c_rank, c_n, c_is_best, c_margin, c_other_best, s_rank,
  s_max, s_gap, s_n05, s_sum, s_n, s_sum_best, p1). Early stopping at 257-398 rounds.
- **Stage-2 OOF F0.5 = 0.9735** (o2o + expected-F; singleton 0.978, non-singleton 0.973);
  global threshold 0.70: 0.9732 (singleton 0.987). Gain over stage 1: +0.0044.
- Gain importance: stage 1 — r_p 48%, a_tset 13%, num_only2 9%; stage 2 — c_margin 69%, p1 29%.
- **Cross-country** (stage 1+2 trained on one country, threshold from its OOF): US->India 0.850,
  India->US 0.952 (mean 0.901). US->India loss: Indic scripts / landmark addresses absent from US.
- Test: 5.63M matches (baseline 7.94M); predicted-count distribution per country nearly identical
  (empty: France 5.9%, India 6.1%, US 5.9%).

## Phase 8 — decoding / proxy validation (src/decode.py, src/proxy_val.py, src/run_proxy.py)
- one-to-one helps slightly for threshold decoding; expected-F (MC 500 samples, k<=10) best on OOF.
- Label-free composition comparison train vs test (cache/proxy.log): US/India test ~ train (slightly
  more candidates/S1, no extra weak-best S1s). Injecting singletons or removing S1s only increases the
  distance to test -> best proxy = unmodified train, so OOF is a fair guide for US/India.
- **France is the outlier:** 24 cands/S1 and 4.4 competing S1s per candidate (US/India ~2.3-2.6) ->
  very generic names. Explains the baseline LB drop (implied France F ~0.15 for the baseline).

## Phase 9 — France robustness
- See notes_france.md: normaliser handles legal forms / street types / accents; gaps: départements not
  mapped to regions, EI/Ets/Cie legal forms. Manual inspection of French predictions: correct.
- No code path depends on country values (country only partitions blocking; validation code aside).
  No country-derived features exist, so the Phase-9 ablation is moot.

## Phase 10 — final build (in progress)
- Documentation_template.md and README filled. Reproducibility: fresh `git clone` into
  ../repro_clone, caches for normalisation/blocking/features hardlinked, model stage re-run from the
  entrypoint; outputs to be diffed against output_model/.
- Reproducibility (fresh clone c524f5a..2b2ce44 code, entrypoint `--mode model`): stage-1 5-fold OOF
  predictions (25.7M) are **bit-identical** to the main run (max abs diff 0.0; best iterations
  800/800/799/800/798 identical). The run was then stopped by the host for low system memory during
  stage 2 (not a pipeline error); remaining stages not yet re-verified.
