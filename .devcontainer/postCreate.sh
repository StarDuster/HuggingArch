#!/usr/bin/env bash
set -euo pipefail
cd /workspace/HuggingArch
uv sync --dev
echo "[postCreate] uv sync done. .venv at /workspace/.venv"
