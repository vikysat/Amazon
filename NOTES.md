# NOTES — Business Entity Resolution (Amazon ML Challenge 2026)

## Status (keep updated)
- **Current phase:** 0/1 done → Phase 2–4 in progress (normalisation, blocking, baseline)
- **Best OOF F0.5:** n/a yet
- **Next step:** blocking + baseline submission

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
