# Model Config Policy

- `templates/` contains safe public templates with placeholders.
- `private/` is for local rendered configs with real keys.

Do not commit real secrets. `private/` is ignored by `.gitignore`.

## Quick start

1. Export env vars:
   - `export DEEPSEEK_API_KEY=...`
   - `export GLM_API_KEY=...`
   - `export OPENAI_API_KEY=...`
2. Generate private configs:
   - `bash scripts/prepare_private_config.sh`
