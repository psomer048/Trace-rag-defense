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
