# MCFT Artifact Manifest

This repository contains the runnable code artifact for the AAAI 2027 submission:

**Mass-Conservation Fragment Tokens for Robust and Auditable MS/MS Spectrum Retrieval**

The artifact is intentionally code-only. It excludes local paper drafts, downloaded MassSpecGym TSV files, intermediate outputs, model checkpoints, and Hugging Face Space run logs.

## Required Environment

```bash
python3 -m pip install -r experiment/requirements.txt
export PYTHONPATH=experiment/src
```

## Data

The experiments expect locally generated MassSpecGym TSV files under:

```text
experiment/data/massspecgym/
```

Example fetch command:

```bash
PYTHONPATH=experiment/src python3 -m eventclock.fetch_massspecgym_rows \
  --out experiment/data/massspecgym/MassSpecGym_rows_25k.tsv \
  --limit 25000 \
  --workers 1 \
  --sleep-seconds 0.2
```

The data directory is ignored by Git.

## Paper-Critical Scripts

These scripts reproduce the main claims in the paper:

- `eventclock.run_massspecgym_retrieval_smoke`: modified cosine, fixed MCFT, and ridge MCFT retrieval baselines.
- `eventclock.run_massspecgym_mcft_transformer`: MCFT-Transformer listwise retrieval experiments.
- `eventclock.run_massspecgym_listwise_linear`: listwise linear MCFT capacity control.
- `eventclock.audit_mcft_evidence_examples`: fragment-witness export.
- `eventclock.audit_mcft_witness_removal`: counterfactual top-witness deletion audit.
- `eventclock.run_mass_conservation_tokens`: synthetic conservation-token sanity test.

Diagnostic scripts for formula-conditioned reranking and delta/cross-adduct probes are retained only to document negative or non-core results. They are not part of the main contribution claims.

## Main Paper Commands

See `experiment/README.md` for full commands matching the manuscript settings.

The main 25k closest-mass stress test uses:

```bash
PYTHONPATH=experiment/src python3 -m eventclock.run_massspecgym_mcft_transformer \
  --tsv experiment/data/massspecgym/MassSpecGym_rows_25k.tsv \
  --out-dir experiment/outputs/massspecgym_25k_mcft_transformer_closest20_hard500 \
  --train-queries 2400 \
  --train-negatives 63 \
  --eval-queries 300 \
  --eval-negatives 500 \
  --query-folds val,test \
  --candidate-folds val,test \
  --negative-strategy closest \
  --negative-window 20 \
  --epochs 10 \
  --batch-size 8 \
  --max-tokens 96 \
  --device cuda
```

