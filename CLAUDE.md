# TASK: Build a winning Business Entity Resolution pipeline (Amazon ML Challenge 2026)

You are building a complete, reproducible ML pipeline for a competition. Work autonomously, phase by phase. After each phase: run it end-to-end, record the metrics, commit to git, and append a short entry to NOTES.md (what you did, what the metric was, what you learned). Always keep a working pipeline that produces valid output. Never break the last good version while experimenting.

## 1. The problem

Three tab-separated sources of business records: dataset/train/train_source{1,2,3}.tsv and dataset/test/test_source{1,2,3}.tsv. Columns in each: entity_id, business_name, business_address, country. The ID prefix (S1-, S2-, S3-) gives the source.

- Source 1 is deduplicated and is the reference.
- For every S1 entity, find all S2/S3 records that refer to the same real-world business. That can be zero, one, or many.
- Ground truth for train is in dataset/train/train_ground_truth.tsv with columns source1_entity_id and matched_entity_ids (comma-separated, empty for no match).
- Always read files with pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).

Expect this noise in names:
- abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited) and legal-suffix inconsistencies
- DBA/trade names
- & vs "and", word-order swaps, typos, transliterations

Expect this noise in addresses:
- Rd/Road, St/Street
- missing PIN/ZIP/state
- landmark references ("Near SBI ATM")
- varying numbering formats and reordered components

Train covers US and India. **Test also contains France, which never appears in train.** Treat country as an open set of string labels. Never hard-code, filter, or one-hot country to {US, India}. Every test S1 entity, including French ones, must appear in the output.

## 2. Metric: macro F-beta with beta=0.5, averaged per S1 entity

For each S1 entity, with predicted set P and true set T:
- If T is empty and P is empty, the score is 1.0.
- If exactly one of T and P is empty, the score is 0.0.
- Otherwise precision = |P∩T|/|P| and recall = |P∩T|/|T|. The score is F = 1.25·prec·rec / (0.25·prec + rec), or 0 if the intersection is empty.

The final score is the mean over all S1 entities. Precision is weighted 2x over recall, and singletons (correct empty predictions) are worth a full 1.0. When uncertain, predicting nothing is often optimal.

## 3. Hard rules (violations mean disqualification or rejection)

- **No external data lookup of any kind.** No geocoding APIs, no business registries, no web data, no entity-resolution services. Only the provided data plus pretrained open models. Hand-written normalization dictionaries (abbreviations, legal suffixes, street types) written from general knowledge are fine.
- **Model licenses:** every ML model used must be MIT or Apache 2.0 licensed and at most 8B parameters.
  - Before using any pretrained model, check its license on its Hugging Face model card.
  - Record the name and license of each model in NOTES.md.
  - LightGBM (MIT), XGBoost (Apache 2.0), scikit-learn (BSD), rapidfuzz (MIT) and faiss (MIT) are fine.
- **Never tune anything on test labels.** None exist. Use test data only for inference and for fitting unsupervised things like TF-IDF vocabularies.

## 4. Required outputs (both tab-separated, written to output/)

**output/matching_results.tsv**
- Header: source1_entity_id<TAB>matched_entity_ids
- Exactly one row per test S1 entity, including those with no matches (empty second column).
- IDs are comma-separated with no spaces and no quoting.
- No duplicates within a list. Only S2-/S3- IDs that exist in the test files.

**output/candidate_pairs.tsv**
- Header: source1_entity_id<TAB>candidate_entity_ids
- Same rules as above.
- This must be the exact candidate set the final model scores: the last filtering stage before inference, not an earlier, broader blocking pass.
- Every matched ID must also appear in that entity's candidates.

Write both files with the csv module or pandas using quoting=csv.QUOTE_NONE. After writing, always run:

    python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test

It must print PASS.

## 5. Required repo layout (this is the final submission package)

    code/business_entity_resolution/
      src/            # all source, as modules
      README.md       # exact reproduce steps: data -> blocking -> matching -> output
      requirements.txt  # pinned versions (pip freeze of what you actually used)
    output/
    Documentation_template.md  # fill in the provided template at the end

Requirements for the code:
- One entrypoint regenerates everything: `python src/run_pipeline.py --data-dir <path to dataset> --out-dir <path to output>`.
- Set seeds everywhere. Cache intermediate artifacts (parquet) so reruns are fast.
- Add docstrings and comments to every function; the code gets reviewed.

## 6. Phases

### Phase 0: Setup and EDA (write findings to NOTES.md)
- git init, create the layout, set up a venv.
- Report row counts per file and country, the singleton rate in train, and the distribution of match counts per S1 entity.
- Report the S2 vs S3 share of matches.
- **Check whether any S2/S3 ID appears under more than one S1 entity in the ground truth.** If none does, the "each S2/S3 record belongs to at most one S1" constraint holds, and you should exploit it later.
- Check how often country agrees between matched records.
- Print 30 random matched pairs and 10 hard-looking ones (low string similarity), and describe the noise patterns you see per country.
- Look at test-only French records to understand their format (addresses, postcodes, legal forms).

### Phase 1: Scorer and validation harness
- src/metrics.py implements the metric above exactly. Unit-test it against the worked example in the problem statement: predicted [A,B,C], truth [A,C] gives 0.714.
- Validation uses 5-fold GroupKFold over S1 entities.
  - Keep ALL S2/S3 records in the candidate pool, so distractors look like the test set.
  - Evaluate only the held-out S1 entities.
  - All tuned choices (thresholds, decoding) come from out-of-fold predictions.
- Also add cross-country validation: train on US and evaluate on India, then the reverse. This is the proxy for how well the pipeline generalizes to France. Report it every time.
- Every experiment appends a row to experiments.csv with: timestamp, git hash, description, blocking recall, mean candidates per S1, OOF F0.5, singleton accuracy, non-singleton F0.5, and cross-country F0.5.

### Phase 2: Normalization (src/normalize.py)
The normalizer must be language-aware but country-agnostic in structure.
- Unicode NFKD, strip accents, lowercase, unify punctuation, "&" to "and".
- Expand or canonicalize abbreviations for US, India, and France:
  - Legal forms: pvt/private, ltd/limited, llc, inc, corp, co, llp, and France's sarl, sas, sasu, sa, eurl, sci, "societe".
  - Street types: st/street, rd/road, ave/avenue, blvd/boulevard, and France's rue, av, bd, bld, chemin, allee, place, "cedex".
- Produce these fields:
  - name_core: legal suffixes removed
  - name_legal: the legal form alone
  - name_tokens
  - addr_norm
  - addr_numbers: house or street numbers
  - postcode: a 6-digit PIN, a 5-digit US ZIP or French code, detected by regex and never gated on country
  - city: best effort from the last address tokens
  - addr_no_landmark: address with "near ...", "opp ...", "behind ..." phrases removed
- Keep the raw strings too; models may use them.

### Phase 3: Blocking (src/blocking.py)
Union several blockers. Tag each candidate pair with which blockers produced it; these tags become features.
- **(a) Name character n-grams:** TF-IDF (char_wb 3–4 grams) on name_core, cosine top-K nearest neighbors from S1 to S2 and S1 to S3.
- **(b) Name plus address:** the same, on name_core + addr_norm.
- **(c) Rare-token blocking:** an inverted index on name tokens weighted by IDF. Share at least one rare token (IDF above a threshold), with a cap on block size.
- **(d) Postcode:** exact postcode match combined with some name similarity.
- **(e) Dense embeddings:** a multilingual sentence-embedding model on "name | address", top-K via faiss or sklearn. Candidates: intfloat/multilingual-e5-small or -base, or BAAI/bge-m3. Check the license first; use a GPU if available, otherwise the small model.

Tune K per blocker on train OOF:
- Report pair recall (the fraction of true pairs captured), per-entity recall ceiling, mean candidates per S1, and reduction ratio.
- Target at least 98% pair recall with as few candidates as possible.
- Don't block strictly on country. Cross-country pairs should get a feature, not be excluded, unless EDA shows country always agrees.

### Phase 4: Baseline end-to-end submission (DO THIS EARLY)
- Score candidates with a simple average of name and address similarity, pick one global threshold on OOF, and write both outputs.
- Run the validator and commit. That is the safety-net submission.
- Tell me the OOF F0.5 and that output/matching_results.tsv is ready to upload.

### Phase 5: Stage-1 pairwise model (src/features.py, src/model.py)
For every candidate pair, compute these features:
- **Name:**
  - rapidfuzz ratio, partial_ratio, token_sort_ratio, token_set_ratio, Jaro-Winkler
  - TF-IDF word and character cosine
  - IDF-weighted token Jaccard, and the IDF sum of shared vs unshared tokens
  - acronym match (the initials of one name equal a token of the other)
  - first-token match
  - legal-form agree, conflict, or missing
  - length ratio
  - numeric tokens in names agree or conflict
- **Address:**
  - fuzzy ratios on addr_norm and addr_no_landmark
  - postcode equal, conflict, or missing on either side
  - house-number overlap or conflict
  - city match
  - street-token overlap
  - address missing or short flags
- **Other:**
  - embedding cosine
  - blocker tags and per-blocker rank or score
  - source (S2 or S3)
  - country equal flag (as a generic equality, never country identity)
  - string lengths

Train LightGBM (binary) with GroupKFold over S1 IDs to produce OOF probabilities for all train candidates. Use sensible regularization and early stopping on an inner split. Calibrate if needed (isotonic on OOF) and report the calibration curve summary.

### Phase 6: Stage-2 relational features (big expected gain)
Using stage-1 OOF probabilities (never in-fold, to avoid leakage):
- **For each candidate record c of S1 entity s:**
  - The rank of s among all S1 entities that have c as a candidate, and whether s is c's best S1.
  - The margin between prob(s,c) and c's second-best S1.
  - How many S1s have c as a candidate.
- **For each S1 entity s:**
  - The rank of c among s's candidates, the max prob, the gap to the max, and the number of candidates above 0.5.
- **S2–S3 consistency:** max over other candidates c' of s of (similarity(c, c') × prob(s, c')). Evidence that c agrees with another strong match of s.

Train a stage-2 LightGBM on stage-1 features plus these, with the same GroupKFold, and report the gain.

### Phase 7: Neural cross-encoder (if a GPU is available; otherwise skip and note it)
- Fine-tune a multilingual cross-encoder on pairs formatted as "name | address" [SEP] "name | address". Train on the train candidates with the same folds, and produce OOF scores for every pair.
  - Candidates: microsoft/mdeberta-v3-base (MIT), xlm-roberta-base (MIT), or BAAI/bge-reranker-v2-m3. Verify each license.
- Add its score as a feature to the stage-2 model. Report the gain overall and on cross-country validation.

### Phase 8: Decoding and post-processing (src/decode.py)
Tune everything on OOF and keep only what improves OOF F0.5.

1. **One-to-one constraint:** if Phase 0 confirmed it, each S2/S3 record is assigned only to its highest-probability S1 and dropped from the others.
2. **Expected-F0.5 decoding per S1 entity:**
   - Sort candidates by calibrated probability.
   - For k = 0..min(n, 10), estimate the expected F0.5 of predicting the top-k. Monte Carlo with about 500 samples of Bernoulli(p_i) outcomes is fine; vectorize it.
   - k=0 scores 1.0 only when no candidate is true. Account for true matches that blocking missed, estimated from OOF.
   - Pick the k with the highest expected score.
3. Compare against the baselines: a single global threshold, and a threshold plus a "max-prob must exceed t0" rule for non-empty predictions.
4. Report which decoding wins, and the singleton accuracy vs the non-singleton F0.5 tradeoff.

### Phase 9: Robustness for France
- Review the normalization output on test French records; spot-check 30.
- Confirm that no code path depends on country values from train.
- Compare cross-country validation for (a) the full model and (b) a model without any country-derived features, and choose the more robust one if the in-domain loss is small.
- Print the distribution of predicted match counts on test per country. If France looks wildly different from US/India (for example, almost everything empty or everything matched), investigate and report before finalizing.

### Phase 10: Final build
- Retrain the chosen models on all train data and run inference on test. Write both outputs, run the validator (must print PASS), and commit.
- README with exact commands, hardware used, and approximate runtime.
- Pinned requirements.txt.
- Fill in Documentation_template.md with:
  - methodology
  - blocking strategy with recall and reduction-ratio numbers
  - feature engineering
  - model architecture
  - decoding
  - validation scheme and cross-country results
  - models used with their licenses
  - an explicit statement that no external data or lookups were used
- Verify reproducibility by running the entrypoint from a fresh clone into a temporary directory and diffing the outputs.

## 7. Working style
- Check dataset sizes first and pick algorithms that fit the hardware. Log time and memory for each stage.
- Prefer fast, vectorized code (rapidfuzz.process.cdist, sparse matrix ops, batched embeddings).
- After each phase, give me a 5-line summary: what changed, OOF F0.5, cross-country F0.5, blocking recall, and the next step.
- If a choice has a big tradeoff (for example, GPU time vs gain), state it and pick the option with the better expected OOF F0.5 per hour of work.
- If you are uncertain whether something violates the fair-play rules, stop and ask me.

## 8. Extra instructions
- Deadline: Sep 27, 2026, 11:59 PM IST. My hardware: [FILL IN: laptop/instance, CPU cores, RAM].
- No GPU for now. Skip Phase 7 unless I say otherwise, and note the skip in NOTES.md and the README. Use the small embedding model (multilingual-e5-small) on CPU in Phase 3.
- Priority: get Phase 4 (baseline, validator PASS) done as fast as possible, before polishing earlier phases.
- After every phase that improves OOF F0.5, copy both output files into submissions/<phase>_<score>/ and run the validator on them.
- There is a separate format-check portal that doesn't count against submissions. Remind me to use it for new outputs. Only recommend a real leaderboard submission when OOF F0.5 has clearly improved over the last submitted version.
- Keep NOTES.md up to date so work can resume after a session restart: current phase, best score so far, next step.
