#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[Info] Generate private config files in model_configs/private/"

if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
  python scripts/render_model_config.py \
    --template model_configs/templates/deepseek-chat_config.template.json \
    --output model_configs/private/deepseek-chat.json \
    --api-key-env DEEPSEEK_API_KEY
fi

if [[ -n "${GLM_API_KEY:-}" ]]; then
  python scripts/render_model_config.py \
    --template model_configs/templates/glm-4.5-air_config.template.json \
    --output model_configs/private/glm-4.5-air.json \
    --api-key-env GLM_API_KEY
fi

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  python scripts/render_model_config.py \
    --template model_configs/templates/gpt-4o-mini_config.template.json \
    --output model_configs/private/gpt-4o-mini.json \
    --api-key-env OPENAI_API_KEY
fi

echo "[Done] Private configs prepared."
