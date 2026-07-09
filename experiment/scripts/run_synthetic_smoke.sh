#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHONPATH=experiment/src python3 -m eventclock.train \
  --config experiment/configs/synthetic_eventclock.yaml \
  --set train.epochs=1 \
  --set dataset.params.splits.train=128 \
  --set dataset.params.splits.val=64 \
  --set dataset.params.splits.test=64 \
  --set eval.max_evidence_batches=1 \
  --set output_dir=\"experiment/outputs/smoke_eventclock\"

