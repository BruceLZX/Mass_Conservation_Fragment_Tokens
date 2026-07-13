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

Controlled low-capacity retrieval baseline:

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

Overlap-hard fragment negatives:

```bash
PYTHONPATH=experiment/src python3 -m eventclock.run_massspecgym_retrieval_smoke \
  --tsv experiment/data/massspecgym/MassSpecGym_rows_25k.tsv \
  --out-dir experiment/outputs/massspecgym_25k_retrieval_valtest_to_valtest_overlap_hard500_peakset_fair \
  --num-queries 300 \
  --num-negatives 500 \
  --query-folds val,test \
  --candidate-folds val,test \
  --negative-strategy overlap \
  --negative-window 120 \
  --learned-pairs 20000 \
  --seeds 0,1,2
```

## Conservation-Token Transformer

The transformer branch upgrades MCFT from a ridge scorer over aggregate features to a listwise neural scorer over sparse fragment-pair evidence tokens. Each query-candidate pair is represented as the top conserved fragment correspondences, and training uses cross-entropy over candidate lists.

Small 10k random hard-negative run:

```bash
PYTHONPATH=experiment/src python3 -m eventclock.run_massspecgym_mcft_transformer \
  --tsv experiment/data/massspecgym/MassSpecGym_rows_10k.tsv \
  --out-dir experiment/outputs/massspecgym_10k_mcft_transformer_random_hard500 \
  --train-queries 1200 \
  --train-negatives 63 \
  --eval-queries 300 \
  --eval-negatives 500 \
  --query-folds val,test \
  --candidate-folds val,test \
  --epochs 8 \
  --batch-size 8 \
  --max-tokens 96 \
  --device cuda
```

Closest-mass stress test:

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

The script also reports modified cosine, fixed zero-shift MCFT, and ridge MCFT on the same evaluation queries.

## Listwise Linear Control

This control uses the same train-list construction as MCFT-Transformer but replaces token attention with a linear softmax scorer over aggregated conservation-token statistics.

```bash
PYTHONPATH=experiment/src python3 -m eventclock.run_massspecgym_listwise_linear \
  --tsv experiment/data/massspecgym/MassSpecGym_rows_25k.tsv \
  --out-dir experiment/outputs/massspecgym_25k_listwise_linear_mcft_closest20_hard500 \
  --train-queries 3000 \
  --train-negatives 63 \
  --eval-queries 300 \
  --eval-negatives 500 \
  --query-folds val,test \
  --candidate-folds val,test \
  --negative-strategy closest \
  --negative-window 20 \
  --epochs 80 \
  --seeds 0,1,2
```

## MLP Statistics Control

This control uses the same aggregated MCFT statistics as the listwise linear control but replaces the linear scorer with a one-hidden-layer MLP. It tests whether the transformer gain can be explained by generic nonlinear capacity over pooled statistics.

```bash
PYTHONPATH=experiment/src python3 -m eventclock.run_massspecgym_mlp_stats_control \
  --tsv experiment/data/massspecgym/MassSpecGym_rows_25k.tsv \
  --out-dir experiment/outputs/massspecgym_25k_mlp_stats_mcft_closest20_hard500 \
  --train-queries 3000 \
  --train-negatives 63 \
  --eval-queries 300 \
  --eval-negatives 500 \
  --query-folds val,test \
  --candidate-folds val,test \
  --negative-strategy closest \
  --negative-window 20 \
  --epochs 80 \
  --width 96 \
  --seeds 0,1,2
```

The same control can be run on the overlap-hard lists by changing the output directory and using `--negative-strategy overlap --negative-window 120`.

## Raw-Token Sum-Pooling MLP Control

This control uses the same raw MCFT token constructor as the transformer, then applies invariant sum/mean/max/min pooling before a one-hidden-layer MLP. By default it drops precursor-derived token fields, so it tests whether simple pooling over fragment-witness tokens can explain the learned scorer without using precursor-gap metadata.

```bash
PYTHONPATH=experiment/src python3 -m eventclock.run_massspecgym_sum_pool_mlp \
  --tsv experiment/data/massspecgym/MassSpecGym_rows_25k.tsv \
  --out-dir experiment/outputs/massspecgym_25k_sum_pool_mlp_no_precursor_closest20_hard500 \
  --train-queries 3000 \
  --train-negatives 63 \
  --eval-queries 300 \
  --eval-negatives 500 \
  --query-folds val,test \
  --candidate-folds val,test \
  --negative-strategy closest \
  --negative-window 20 \
  --epochs 80 \
  --width 96 \
  --seeds 0,1,2
```

## Hard-Negative Diagnostics

This audit compares random, closest-mass, and overlap-hard 500-negative lists on the same parsed 25k sample. It reports precursor-gap statistics and query-level maximum negative overlap under modified cosine and zero-shift MCFT.

```bash
PYTHONPATH=experiment/src python3 -m eventclock.audit_massspecgym_hardness \
  --tsv experiment/data/massspecgym/MassSpecGym_rows_25k.tsv \
  --out-dir experiment/outputs/massspecgym_25k_hardness_audit_random_hard500 \
  --num-queries 300 \
  --num-negatives 500 \
  --query-folds val,test \
  --candidate-folds val,test \
  --negative-strategy random \
  --negative-window 120 \
  --seeds 0,1,2
```

```bash
PYTHONPATH=experiment/src python3 -m eventclock.audit_massspecgym_hardness \
  --tsv experiment/data/massspecgym/MassSpecGym_rows_25k.tsv \
  --out-dir experiment/outputs/massspecgym_25k_hardness_audit_closest20_hard500 \
  --num-queries 300 \
  --num-negatives 500 \
  --query-folds val,test \
  --candidate-folds val,test \
  --negative-strategy closest \
  --negative-window 20 \
  --seeds 0,1,2
```

```bash
PYTHONPATH=experiment/src python3 -m eventclock.audit_massspecgym_hardness \
  --tsv experiment/data/massspecgym/MassSpecGym_rows_25k.tsv \
  --out-dir experiment/outputs/massspecgym_25k_hardness_audit_overlap_hard500 \
  --num-queries 300 \
  --num-negatives 500 \
  --query-folds val,test \
  --candidate-folds val,test \
  --negative-strategy overlap \
  --negative-window 120 \
  --seeds 0,1,2
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

Counterfactual top-witness removal audit:

```bash
PYTHONPATH=experiment/src python3 -m eventclock.audit_mcft_witness_removal \
  --tsv experiment/data/massspecgym/MassSpecGym_rows_10k.tsv \
  --out-dir experiment/outputs/massspecgym_10k_witness_removal_audit \
  --num-queries 300 \
  --num-negatives 500 \
  --query-folds val,test \
  --candidate-folds val,test \
  --learned-pairs 12000 \
  --seed 0 \
  --top-k 3 \
  --random-trials 10
```

This audit deletes the top positive-candidate conservation witness peaks and compares the resulting rank and score drop against random peak deletion.

Transformer checkpoint witness-removal audit:

```bash
PYTHONPATH=experiment/src python3 -m eventclock.audit_mcft_transformer_witness_removal \
  --checkpoint experiment/results/space_runs/mcft_transformer_20260712T120958Z/10k_random_hard500_seed0/best.pt \
  --tsv experiment/data/massspecgym/MassSpecGym_rows_10k.tsv \
  --out-dir experiment/outputs/massspecgym_10k_transformer_witness_removal_seed0 \
  --eval-queries 300 \
  --eval-negatives 500 \
  --query-folds val,test \
  --candidate-folds val,test \
  --negative-strategy random \
  --negative-window 120 \
  --seed 0 \
  --device cuda
```

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
