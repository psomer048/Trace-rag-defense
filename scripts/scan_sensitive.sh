#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python - <<'PY'
import re
from pathlib import Path

TEXT_EXTS = {".py", ".md", ".json", ".yml", ".yaml", ".sh", ".txt", ".toml", ".ini", ".cfg"}
RULES = [
    ("API key patterns", re.compile(r"sk-[A-Za-z0-9_-]{10,}|hf_[A-Za-z0-9]{20,}")),
    ("Generic secret keywords", re.compile(r"(api[_-]?key|access[_-]?token|secret[_-]?key|password\s*=)", re.IGNORECASE)),
    ("Personal absolute paths", re.compile(r"/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|C:\\Users\\")),
]

def files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in TEXT_EXTS:
            continue
        if any(part in {".git", "__pycache__", ".venv", "venv"} for part in p.parts):
            continue
        yield p

for title, pattern in RULES:
    print(f"[Scan] {title}")
    n = 0
    for p in files(Path(".")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                print(f"{p}:{i}:{line.strip()}")
                n += 1
    if n == 0:
        print("(no matches)")
    print()

print("[Done] Review matches above before publishing.")
PY
