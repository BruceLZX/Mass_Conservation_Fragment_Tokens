# MCFT Experiments

This folder contains experiment code for **Mass-Conservation Fragment Tokens (MCFT)**.

MCFT scores a pair of MS/MS spectra by exposing sparse fragment correspondences as evidence tokens. The current reliable task is same-molecule spectrum-spectrum retrieval under hard negatives. Transformation discovery, exact-formula reranking, and cross-adduct reasoning are included only as diagnostic scripts and should not be treated as main claims unless later results improve.

## Environment

```bash
python3 -m pip install -r experiment/requirements.txt
export PYTHONPATH=experiment/src
```

## Data

The scripts expect locally parsed MassSpecGym TSV files under:

```text
experiment/data/massspecgym/
```

These data files are intentionally ignored by Git because they are generated/downloaded artifacts.

Fetch rows through the Hugging Face dataset-server API:

```bash
PYTHONPATH=experiment/src python3 -m eventclock.fetch_massspecgym_rows \
  --out experiment/data/massspecgym/MassSpecGym_rows_10k.tsv \
  --limit 10000 \
  --workers 1 \
  --sleep-seconds 0.2
```

## Main Retrieval Runs

Random hard negatives:

```bash
PYTHONPATH=experiment/src python3 -m eventclock.run_massspecgym_retrieval_smoke \
  --tsv experiment/data/massspecgym/MassSpecGym_rows_10k.tsv \
  --out-dir experiment/outputs/massspecgym_10k_retrieval_valtest_to_valtest_hard500_peakset_fair \
  --num-queries 300 \
  --num-negatives 500 \
  --query-folds val,test \
  --candidate-folds val,test \
  --learned-pairs 12000 \
  --seeds 0,1,2,3,4
```

Closest-mass hard negatives:

```bash
PYTHONPATH=experiment/src python3 -m eventclock.run_massspecgym_retrieval_smoke \
  --tsv experiment/data/massspecgym/MassSpecGym_rows_25k.tsv \
  --out-dir experiment/outputs/massspecgym_25k_retrieval_valtest_to_valtest_closest20_hard500_peakset_fair \
  --num-queries 300 \
  --num-negatives 500 \
  --query-folds val,test \
  --candidate-folds val,test \
  --negative-strategy closest \
  --negative-window 20 \
  --learned-pairs 20000 \
  --seeds 0,1,2,3,4
```

## Evidence Audit

```bash
PYTHONPATH=experiment/src python3 -m eventclock.audit_mcft_evidence_examples \
  --tsv experiment/data/massspecgym/MassSpecGym_rows_10k.tsv \
  --out-dir experiment/outputs/massspecgym_10k_evidence_examples \
  --num-queries 300 \
  --num-negatives 500 \
  --learned-pairs 12000
```

The audit exports query/candidate IDs, ranks, scores, and top matched fragment pairs.

## Diagnostic Branches

These scripts are retained for reproducibility and early stopping:

- `eventclock.run_massspecgym_formula_conditioned_retrieval`: exact-formula/isomer reranking diagnostic. Current results are not strong enough for the main paper.
- `eventclock.run_massspecgym_delta_retrieval`: formula-delta diagnostic. Current +O setting is shortcut-dominated by precursor mass.
- `eventclock.run_mass_conservation_tokens`: synthetic conservation-token stress test.

## Key Metrics

- `hit@1`: whether the positive spectrum is ranked first.
- `mrr`: reciprocal-rank average.
- `modified_cosine_*`: zero/precursor-shift spectral matching baseline.
- `mcft_zero_shift_*`: fixed zero-shift fragment conservation score.
- `learned_mcft_pair_*`: learned sparse conservation-token scorer.
