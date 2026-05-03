# Poison Scorer Model Layout

## Active scorer

Use this path in new experiments:

- `results/models/ACTIVE_POISON_SCORER.pkl`

Current active target:

- `results/models/poison_rf_v2_bioasq_w134_clean_answerhit_gs4.pkl`

Training provenance:

- This active scorer was trained on BioASQ-derived **non-main-experiment** data (a non-overlapping split for scorer training), not on PopQA subject_200 main evaluation queries.
- Retraining entrypoint in this release: `train_model.py`

## Compatibility path

The legacy default path is kept only for backward compatibility:

- `src/poison_rf.pkl`

It should point to the active scorer so older commands do not silently fall back to the old legacy model.

## Archived scorers

Non-`gs4` scorer artifacts are moved under:

- `PENDING_FOR_USER/cleanup_round_20260503/results_nonessential/archive_non_gs4/`

These are archived for provenance and replay only, and should not be used as the default scorer in current runs.
