# France robustness review (Phase 9)

Spot check: 10 random French records per source (S1/S2/S3) through `normalize.py`, plus manual
inspection of the stage-2 model's ranked candidates for 6 random French S1 entities.

## What works
- Legal forms SAS/SARL/SASU/EURL/SCI/SA, "S.A.S"/"S.C.I." (initials collapse), "Societe" -> name_legal.
- Street types: Rue/R./R, Boulevard/Bd, Avenue/Av, Impasse/Imp., Allée/All, Chemin, Quai; accents
  stripped; elisions "de l'Emery" -> "lemery"; "N°16"/"#22"/"(30)" -> numbers extracted.
- Regions: Hauts-de-France / Nouvelle-Aquitaine / Pays de la Loire -> hdf / naq / pdl.
- Model behaviour (manual inspection): true duplicates with typos ("Fïls", "Rcubaix", "L'ETER"),
  abbreviations and domain-name variants ("nantesclub.com") get p ~ 1.0; same-name entities with a
  different legal form (SAS vs SCI), an empty address, or a nearby house number (150 vs 154) get p ~ 0.
- Predicted match-count distribution on test France (0:5.9%, 1:8.4%, 2:20.3%, 3:25.3%, 4:20.3%,
  5+:20.4%) matches US / India closely -> no sign of collapse.

## Gaps found (fix in the next full rebuild; each requires re-running normalisation -> blocking ->
## features -> models for reproducibility, ~7 h)
1. **Départements are not mapped to their region.** Addresses alternate between the region
   ("Nouvelle-Aquitaine") and the département ("Gironde"); US has the analogous state name/code
   canonicalised, France does not. Effect: lower address token_set for true French pairs, and the
   best-effort `city` falls back to the département ("city='gironde'").
   Fix: add départements of the regions seen (Nord, Pas-de-Calais, Somme, Aisne, Oise -> hdf;
   Gironde, Landes, Dordogne, Charente(-Maritime), Pyrénées-Atlantiques, Lot-et-Garonne, Deux-Sèvres,
   Haute-Vienne, Corrèze, Creuse -> naq; Loire-Atlantique, Maine-et-Loire, Mayenne, Sarthe, Vendée -> pdl)
   to `REGION_CANON`, and skip region codes when picking `city`. Avoid short ambiguous words
   ("Lot", "Var", "Ain") that could collide with English/Indian tokens.
2. Minor legal forms not recognised: "EI" (entreprise individuelle), "Ets" (établissements),
   "Cie"/"Compagnie" (-> co), "SNC" is present, "SELARL" present.
3. Frequent injected noise tokens "(France)", "& Fils", "& Frères", "Groupe", "Cie", "+ Associés"
   stay in name_core. They appear on both sides of true pairs and in look-alike entities, so they are
   left to the model (IDF-weighted features down-weight them).

## Country independence check
- `grep` of the source: `country` is used only in `blocking.tfidf_block` (loop over the distinct
  country strings present, generic equality) and in validation code (cross-country split, reporting).
  No feature, threshold or dictionary lookup is keyed on a country value; all dictionaries are applied
  to every record.
- Country-derived features: none exist (the country-equal flag would be constant 1 because blocking is
  within country), so the "model without country features" of Phase 9 is the model itself.
