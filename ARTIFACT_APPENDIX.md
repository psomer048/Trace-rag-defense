# Artifact Appendix

## Scope

This artifact contains a minimal but executable reproduction package for the core defense method on PopQA subject_200.

## Included Components Checklist

- Core method code:
  - `main_defense.py`
  - `src/defense.py`
  - `src/prompts_defense.py`
  - `src/poisoning_detection.py`
- Retrieval/evaluation support:
  - `evaluate_beir.py`
  - `src/contriever_src/`
- Main experiment assets:
  - `datasets/benchmarks/popqa_subject_200/`
  - `results/query_manifests/popqa_subject_200.json`
  - `results/adv_targeted_results/popqa.json`
  - `results/beir_results/popqa_subject_200-contriever.json`
- Poison scorer model + retraining script:
  - `results/models/ACTIVE_POISON_SCORER.pkl`
  - `results/models/poison_rf_v2_bioasq_w134_clean_answerhit_gs4.pkl`
  - `train_model.py`
- Environment:
  - `requirements.txt`
  - `environment.yml`

## Hardware

- Main runs use a single GPU for efficient model inference.

## Expected Runtime (small sanity run)

- First 5-10 PopQA subject_200 queries: typically a few minutes, depending on API latency.

## Determinism

- Seed is set in the main pipeline (`--seed`).
- API-based model responses may still have non-deterministic variance.
