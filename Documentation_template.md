# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** September 2026

---

## 1. Executive Summary
We resolve every Source-1 business to its Source-2/3 records with a *blocking → two-stage gradient
boosting → constrained decoding* pipeline. A language-aware, country-agnostic normaliser (including a
hand-written Indic-script transliterator and a phonetic "consonant skeleton") feeds a pool-side sparse
TF-IDF blocker; a pairwise LightGBM is followed by a *relational* LightGBM that sees how strongly each
candidate record is claimed by competing S1 entities, and decoding enforces the one-to-one property of
S2/S3 records. Out-of-fold macro F0.5 on train is **0.9735** (baseline 0.7335).

---

## 2. Methodology

### 2.1 Problem Analysis
EDA on train (2.21M S1, 5.03M S2, 5.29M S3; test 1.73M / 4.89M / 5.08M):
- **Match structure:** 5.6% of S1 entities are singletons; most have 2–6 matches (mean ≈ 3.5), split
  48% S2 / 52% S3. **No S2/S3 record is matched to more than one S1** (0 of 7.64M) → each pool record
  belongs to at most one S1. 26% of pool records match nothing (distractors).
- **Country:** matched pairs agree on country in 100% of cases. Test adds France (259k S1), unseen in
  train; we treat country only as an opaque string used for within-country blocking.
- **Name noise:** abbreviations and legal-suffix add/drop/bracketing ("[LLC]", "L.L.C.", Pvt/Private),
  typos and leetspeak ("Pr0jects", "5terling"), word-order swaps, trade names ("F/K/A", "formerly known
  as"), domain names ("holymethodistchurch.com"), and — for ~5–10% of true pairs — a completely
  different name: an Indic-script rendering (9.4% of S2 names, 5.3% of S3 names are non-Latin:
  Devanagari, Bengali, Telugu, Tamil, Kannada, Malayalam, Gujarati, Gurmukhi, Oriya) or a random
  pseudo-word ("Nylaquo"). For those, only the address links the records.
- **Address noise:** St/Street, Rd/Road, state full name vs code, native-script state names, missing
  PIN/ZIP/state, truncated addresses, landmark phrases ("Nr.", "Opp"), "null"/"N/A" fillers,
  component reordering, leading zeros and edited house numbers; 3.3% of pool addresses are empty.
- **France (test only):** legal forms SARL/SAS/SASU/EURL/SCI/S.A.S, "(France)"/"& Fils" insertions,
  Rue/R./Bd/Av abbreviations, region vs département swaps. Names are very generic: a French candidate
  is on average proposed for 4.4 S1 entities (US/India ≈ 2.3–2.6), so false merges are the main risk.

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage classifier (pairwise + relational) + constrained decoding  
**Core Innovation:** (1) pool-side blocking that exploits the one-record-one-owner structure;
(2) a stage-2 model on out-of-fold stage-1 probabilities with *competition* features (the margin of a
candidate's probability over its best rival S1 carries ~70% of the stage-2 gain); (3) script- and
country-agnostic normalisation (Indic → Latin transliteration + phonetic skeletons) so that the unseen
country needs no special code.

---

## 3. Candidate Generation (Blocking)
- **Normalisation** (`normalize.py`): NFKD + accent stripping, Indic transliteration (all Indic Unicode
  blocks mapped by offset onto one Devanagari→Latin table with schwa handling), legal-form / street-type
  / region canonicalisation for US, India and France, ordinal words → digits, leading zeros stripped,
  leetspeak undone, landmark phrases removed. Derived fields: name_core, name_legal, name_skel
  (phonetic consonant skeleton: "digital infotech" and "डिजिटल इंफोटेक" → "djtl inptk"), addr_norm,
  addr_no_landmark, addr_numbers, postcode (regex, never country-gated), city.
- **Blocking keys used:** one sparse TF-IDF (sublinear tf, L2) per country over a token bag of
  name-skeleton unigrams + bigrams, address unigrams + bigrams and character 4-grams of the space-free
  name; tokens with document frequency > 1000 are dropped from the sparse product. Each S2/S3 record
  retrieves its top-6 S1 entities by cosine (chunked sparse matrix product, 14 processes). The pool→S1
  direction was chosen because each pool record has at most one owner; at equal candidate budget it
  matched the recall of adding an S1→pool pass at a fraction of the cost.
- **Final candidate filter** (the set scored by the model and written to candidate_pairs.tsv): keep a
  pair if its pool→S1 rank ≤ 1 or its cosine ≥ 0.4.
- **Candidate pairs generated:** train 25.7M (11.7 per S1), test 26.2M (15.1 per S1).
  Raw blocking produced 61.9M / 59.8M pairs.
- **How we ensured true matches were not lost:** recall measured on train with full-size pools
  (all distractors present): raw pair recall 96.8%, after the filter **96.0% pair recall**, mean
  per-entity recall 96.0%, 88.3% of entities have all their matches in the candidate set;
  **reduction ratio 0.999999**. Ablations on a 20k-entity sample: character 4-grams +1.5–2 pp recall;
  bigrams made the product ~10× faster than unigrams alone. Remaining misses are mostly generic names
  with empty or truncated addresses and pseudo-word names with edited addresses.

---

## 4. Matching Model

**Features used (57):**
- Name: rapidfuzz ratio / partial_ratio / token_sort / token_set / Jaro-Winkler on name_core; ratio and
  token_set on phonetic skeletons; token_set on raw names; IDF-weighted token Jaccard, IDF mass of shared
  and unshared tokens, shared-token count, skeleton Jaccard; acronym match; first-token match;
  legal-form agree / conflict / missing; length ratio; numeric-token agree / conflict.
- Address: ratio / token_set / token_sort / partial on addr_norm, token_set on landmark-free address;
  postcode agree / conflict / missing; house-number overlap, Jaccard, first-number agree/conflict,
  unmatched numbers on each side; city similarity; street-word Jaccard and overlap; lengths; empty flag.
- Other: blocker cosine and rank, source (S2/S3), string lengths. Country is used only as a blocking
  partition (generic string equality); no country identity feature exists.
- Stage-2 relational features (from out-of-fold stage-1 probabilities): per candidate record — rank of
  this S1 among the S1s proposing it, is-best flag, margin to the best rival S1, rival's probability,
  number of proposing S1s; per S1 — rank of the candidate, max probability, gap to max, number of
  candidates above 0.5, probability sum, candidate count, probability mass of candidates it wins.

**Model type:** two LightGBM binary classifiers (MIT). Stage 1 on pairwise features; stage 2 on
pairwise + relational features. lr 0.1, 127 leaves, min_data_in_leaf 200, feature/bagging fraction 0.8,
λ2 = 10, early stopping on an inner 10% split. 5-fold GroupKFold over S1 entities; relational features
for training rows always come from out-of-fold stage-1 predictions. Final models are refit on all
train (a 37.5% random subset of S1 entities for memory) with the mean best iteration.
Top gain: stage 1 — blocker rank 48%, address token_set 13%, unmatched house numbers 9%; stage 2 —
margin to best rival S1 69%, stage-1 probability 29%.

**Threshold selection method:** decoding tuned only on out-of-fold predictions. (1) One-to-one: every
S2/S3 record is kept only for its highest-probability S1. (2) Per S1 entity, expected-F0.5 decoding:
candidates sorted by probability; for k = 0..10 the expected F0.5 of predicting the top-k is estimated
by Monte-Carlo (500 Bernoulli samples, vectorised); the best k is chosen (k = 0 is worth 1.0 only if no
candidate is true). Compared with a global threshold (best t = 0.70) — see Section 5.

---

## 5. Results & Error Analysis

| system | OOF F0.5 | singleton acc. | non-singleton F0.5 |
|---|---|---|---|
| baseline: mean(name, address token_set) + one-to-one + threshold | 0.7335 | 0.100 | 0.771 |
| stage-1 LightGBM, one-to-one + threshold 0.675 | 0.9691 | 0.964 | 0.969 |
| stage-2 LightGBM, global threshold 0.70 | 0.9732 | 0.987 | 0.972 |
| **stage-2 LightGBM, one-to-one + expected-F0.5** | **0.9735** | 0.978 | 0.973 |

- **F_0.5 Score (macro, 5-fold OOF on train):** 0.9735
- **Cross-country validation** (train on one country, threshold from its OOF, evaluate on the other —
  our proxy for the unseen France): US→India 0.850, India→US 0.952. The US→India drop comes from Indic
  scripts and landmark-style addresses that never occur in US data.
- **Leaderboard:** baseline 0.645 public (0.7335 on train). A label-free comparison of train and test
  composition showed US/India test resemble train (no singleton shift); the gap came from France, where
  generic names make string-similarity thresholds merge many different entities. The relational stage-2
  model is designed for exactly this; manual inspection of French predictions shows correct rejection of
  same-name entities that differ in legal form or house number. Model leaderboard score: [TBD].
- **Common false positives (wrong merges):** entities with the same generic name at nearby house
  numbers on the same street ("Val Club Holding SAS" at no. 154 vs "Val Club SAS" at no. 150), and
  same-name entities that differ only in legal form or trade suffix.
- **Common false negatives (missed matches):** records outside the candidate set (≈4% of true pairs:
  generic name + empty/truncated address), pseudo-word replacement names with truncated addresses, and
  script-switched names whose transliteration diverges strongly (Tamil lacks aspirates/voicing).

---

## 6. Conclusion
Careful normalisation plus cheap, recall-oriented blocking made a 12M-record problem tractable on a
laptop, and the two-stage design turned the "each record has one owner" structure into the strongest
signal: the margin over competing S1 entities. Main lessons: measure the train/test composition shift
early (it explained the baseline's leaderboard gap) and prefer country-agnostic mechanisms over
country-specific rules so that an unseen country is handled by the same code path.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/`:
- `run_pipeline.py` — single entrypoint: `python src/run_pipeline.py --data-dir <dataset> --out-dir output`
- `io_utils.py` (TSV I/O, seeds, output writer), `normalize.py` + `prepare.py` (normalisation, cached
  parquet), `blocking.py` (sparse TF-IDF top-k), `stages.py` (cached stages, filter, blocking report),
  `features.py` (pairwise features), `model.py` (stage 1/2 LightGBM, OOF CV, cross-country, inference),
  `decode.py` (one-to-one, thresholds, expected-F0.5), `metrics.py` (competition metric + self-test),
  `proxy_val.py` + `run_proxy.py` (train/test composition analysis), `eda.py`, `explog.py`,
  `repro_test_stages.py`.
- Every run appends to `experiments.csv`; `NOTES.md` has the full experiment log.

**Models and licenses:** LightGBM 4.7.0 (MIT). Libraries: scikit-learn (BSD-3), rapidfuzz (MIT),
pandas / numpy / scipy / pyarrow (BSD / Apache-2.0). No pretrained neural models were used (a
multilingual cross-encoder / dense embedder was skipped: no GPU, and CPU embedding of 22M records was
too slow).

**Fair play:** no external data, APIs, geocoding, business registries or web look-ups of any kind were
used. All normalisation dictionaries (legal forms, street types, US states / Indian states / French
regions, Indic transliteration table) were hand-written from general knowledge. No choice was tuned on
test labels (none exist); test data was used only for inference and for unsupervised statistics
(TF-IDF vocabularies, token IDF, the label-free composition comparison).

### B. Additional Results
- Blocking recall vs budget (train, filter grid): rank 0 or cos ≥ 0.4 → 8.5 cands/S1, 95.4%;
  **rank ≤ 1 or cos ≥ 0.4 → 11.7, 96.0%**; rank ≤ 2 or cos ≥ 0.4 → 15.4, 96.3%; all top-6 → 28.0, 96.8%.
- Test predicted match-count distribution (share of S1 entities with 0/1/2/3/4/5+ matches):
  France 5.9/8.4/20.3/25.3/20.3/20.4%, India 6.1/7.9/19.0/24.3/20.5/23.2%, US 5.9/6.7/18.9/24.9/21.0/22.5%.
- Hardware: Intel i7-13620H (10 cores / 16 threads), 16 GB RAM, no GPU. Runtime from scratch ≈ 7 h
  (normalisation 25 min, blocking ≈ 4 h, features ≈ 2 h, models ≈ 3 h).
