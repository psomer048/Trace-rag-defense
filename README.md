# Artifact for Double-Blind Review

This repository contains an anonymized implementation and evaluation package for the submitted paper.

## Contents

- `src/`: core defense implementation.
- `main_defense.py`: main defense entrypoint.
- `evaluate_beir.py`: retrieval evaluation / cache generation.
- `train_model.py`: poison-scorer training script.
- `datasets/benchmarks/popqa_subject_200/`: bundled small benchmark subset for artifact evaluation.
- `results/query_manifests/popqa_subject_200.json`: query manifest.
- `results/adv_targeted_results/popqa.json`: attack manifest.
- `results/beir_results/popqa_subject_200-contriever.json`: retrieval cache.
- `results/models/ACTIVE_POISON_SCORER.pkl`: active poison scorer checkpoint.

## Environment Setup

### Option A: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate RPD
```

## Private Model Config Setup

```bash
export DEEPSEEK_API_KEY=...
export GLM_API_KEY=...
export OPENAI_API_KEY=...
bash scripts/prepare_private_config.sh
```

Generated private configs are written to `model_configs/private/` and are not intended for public tracking.

## Quick Start (Small Reproduction Run)

```bash
HF_ENDPOINT=https://hf-mirror.com conda run --no-capture-output -n RPD python main_defense.py \
  --model_name glm-4.5-air \
  --model_config_path model_configs/private/glm-4.5-air.json \
  --eval_dataset popqa \
  --split test \
  --data_path datasets/benchmarks/popqa_subject_200 \
  --query_manifest_path results/query_manifests/popqa_subject_200.json \
  --attack_manifest_path results/adv_targeted_results/popqa.json \
  --orig_beir_results results/beir_results/popqa_subject_200-contriever.json \
  --attack_method LM_targeted \
  --adv_per_query 1 \
  --top_k 10 \
  --M 10 \
  --repeat_times 1 \
  --start_index 0 \
  --query_results_dir defense \
  --name artifact_repro_run
```

## Regenerate Retrieval Cache

```bash
python evaluate_beir.py \
  --dataset popqa \
  --split test \
  --data_path datasets/benchmarks/popqa_subject_200 \
  --result_output results/beir_results/popqa_subject_200-contriever.json
```

## Retrain the Poison Scorer

```bash
python train_model.py \
  --data-file poison_training_data.csv \
  --model-file results/models/poison_rf_custom.pkl \
  --schema v2 \
  --mode binary \
  --label-column label
```

## Notes for Review

- This artifact provides a compact executable pipeline for review.
- Full-scale experiments may require substantial API budget.

## Safety Check

```bash
bash scripts/scan_sensitive.sh
```
