# Release Curation Notes

This note summarizes what is currently kept as the active minimal framework and what has been moved out for manual review.

## Active minimal framework (kept)

- Core entrypoints:
  - `main_defense.py`
  - `evaluate_beir.py`
  - `train_model.py`
- Runtime package:
  - `src/` (only modules required by defense/retrieval/training)
- Main experiment data/assets:
  - `datasets/benchmarks/popqa_subject_200/`
  - `results/query_manifests/popqa_subject_200.json`
  - `results/adv_targeted_results/popqa.json`
  - `results/beir_results/popqa_subject_200-contriever.json`
- Poison scorer:
  - `results/models/ACTIVE_POISON_SCORER.pkl`
  - `results/models/poison_rf_v2_bioasq_w134_clean_answerhit_gs4.pkl`
- Config/environment:
  - `requirements.txt`
  - `environment.yml`
  - `model_configs/templates/`
  - `scripts/prepare_private_config.sh`
  - `scripts/render_model_config.py`
  - `scripts/scan_sensitive.sh`

## Moved for user decision

- `PENDING_FOR_USER/cleanup_round_20260503/`
- `PENDING_FOR_USER/created_scripts/`
- `PENDING_FOR_USER/smoke_assets/`
- `PENDING_FOR_USER/reference_experiment_settings/`

Notable interface-only module moved out:
- `src/gt_generator.py` (GT injection path; not used by PopQA main reproduction flow)

## Suggested final deletion candidates (optional)

If you want an even cleaner final public artifact, you can consider removing:
- `PENDING_FOR_USER/smoke_assets/`
- `PENDING_FOR_USER/created_scripts/`
- `PENDING_FOR_USER/cleanup_round_20260503/results_nonessential/`
- `PENDING_FOR_USER/cleanup_round_20260503/misc_unused/`
