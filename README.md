# Mass-Conservation Fragment Tokens

Experiment code for **Mass-Conservation Fragment Tokens (MCFT)**, a sparse fragment-evidence representation for MS/MS spectrum-spectrum retrieval.

The repository intentionally keeps only runnable experiment code and lightweight documentation. Local paper drafts, parsed datasets, output metrics, and Hugging Face Space artifacts are excluded from GitHub.

See `ARTIFACT.md` for the submission artifact manifest and the paper-critical script map.

## Environment

```bash
python3 -m pip install -r experiment/requirements.txt
export PYTHONPATH=experiment/src
```

## Core Experiments

Synthetic conservation-token stress test:

```bash
PYTHONPATH=experiment/src python3 -m eventclock.run_mass_conservation_tokens \
  --out-dir experiment/outputs/mcft_v2_smoke
```

MassSpecGym spectrum-pair retrieval:

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

Fragment-witness audit:

```bash
PYTHONPATH=experiment/src python3 -m eventclock.audit_mcft_evidence_examples \
  --tsv experiment/data/massspecgym/MassSpecGym_rows_10k.tsv \
  --out-dir experiment/outputs/massspecgym_10k_evidence_examples
```

See `experiment/README.md` for the complete experiment map.
