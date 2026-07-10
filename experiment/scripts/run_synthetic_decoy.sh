#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHONPATH=experiment/src python3 -m eventclock.train --config experiment/configs/synthetic_decoy_eventclock.yaml
PYTHONPATH=experiment/src python3 -m eventclock.run_grid --config experiment/configs/synthetic_decoy_grid.yaml
PYTHONPATH=experiment/src python3 -m eventclock.run_grid --config experiment/configs/synthetic_decoy_fixed_patch_grid.yaml
PYTHONPATH=experiment/src python3 -m eventclock.run_grid --config experiment/configs/synthetic_decoy_token_baseline_grid.yaml
PYTHONPATH=experiment/src python3 -m eventclock.summarize_results experiment/outputs --out experiment/outputs/decoy_summary.csv

