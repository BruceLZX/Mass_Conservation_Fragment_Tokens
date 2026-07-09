#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: $0 experiment/configs/<dataset>_npz_eventclock.yaml"
  exit 2
fi
cd "$(dirname "$0")/../.."
PYTHONPATH=experiment/src python3 -m eventclock.train --config "$1"

