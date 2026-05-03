# Scripts

## Active scripts

- `prepare_private_config.sh`
  - Generate `model_configs/private/*.json` from public templates via env vars.
- `render_model_config.py`
  - Helper used by `prepare_private_config.sh`.
- `scan_sensitive.sh`
  - Scan for likely key leaks, secret keywords, and personal absolute paths.

## Training and runs

- Main defense run command is documented in `README.md`.
- Poison scorer retraining entrypoint is `train_model.py` at repo root.

## Moved for manual processing

Per latest cleanup request, previously added demo/smoke scripts were moved to:
- `PENDING_FOR_USER/created_scripts/`

Smoke/demo generated assets and outputs were moved to:
- `PENDING_FOR_USER/smoke_assets/`
