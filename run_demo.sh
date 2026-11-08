#!/usr/bin/env bash
# usage: bash run_demo.sh -- one synthetic scene end-to-end (render -> dff -> scale fit + refine -> metrics + fig). no downloads.
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 code/experiments/run_demo.py
