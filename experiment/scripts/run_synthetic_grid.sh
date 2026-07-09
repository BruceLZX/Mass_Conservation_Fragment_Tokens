#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHONPATH=experiment/src python3 -m eventclock.run_grid --config experiment/configs/synthetic_grid.yaml

