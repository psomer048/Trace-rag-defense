#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a private model config from a public template using env vars."
    )
    parser.add_argument("--template", required=True, help="Template JSON path.")
    parser.add_argument("--output", required=True, help="Rendered JSON output path.")
    parser.add_argument("--api-key-env", required=True, help="Environment variable for API key.")
    parser.add_argument("--index", type=int, default=0, help="api_key_use index (default: 0).")
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"Environment variable '{args.api_key_env}' is empty.")

    template_path = Path(args.template)
    output_path = Path(args.output)
    payload = json.loads(template_path.read_text(encoding="utf-8-sig"))

    payload.setdefault("api_key_info", {})
    payload["api_key_info"]["api_keys"] = [api_key]
    payload["api_key_info"]["api_key_use"] = int(args.index)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] wrote private config: {output_path}")


if __name__ == "__main__":
    main()
