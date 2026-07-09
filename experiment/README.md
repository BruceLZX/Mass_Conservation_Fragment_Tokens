# EventClock Experiments

This folder contains the experiment scaffold for **Evidence-Calibrated EventClock**: a monotone event-time tokenizer trained with task loss plus sufficiency/necessity evidence objectives.

## What Is Covered

- Synthetic sparse-motif sanity check with ground-truth evidence masks.
- Main model: EventClock Transformer.
- Baselines: CNN, fixed-patch Transformer, random token selection, and local-complexity token selection.
- Reviewer-facing checks: token-budget sweeps, fixed-patch sweep, ablations for sufficiency/necessity, deletion/insertion evidence curves, clock-mask overlap, seed sweeps, and robustness perturbations.
- Dataset entry points for PTB-XL, Sleep-EDF, WESAD, and PPG-DaLiA through a shared preprocessed NPZ format.

## Environment

```bash
python3 -m pip install -r experiment/requirements.txt
```

Use the source tree directly:

```bash
export PYTHONPATH=experiment/src
```

## Smoke Test

```bash
experiment/scripts/run_synthetic_smoke.sh
```

The smoke test writes:

- `experiment/outputs/smoke_eventclock/best.pt`
- `experiment/outputs/smoke_eventclock/metrics.json`

## Main Synthetic Runs

```bash
PYTHONPATH=experiment/src python3 -m eventclock.train --config experiment/configs/synthetic_eventclock.yaml
PYTHONPATH=experiment/src python3 -m eventclock.train --config experiment/configs/synthetic_fixed_patch.yaml
PYTHONPATH=experiment/src python3 -m eventclock.run_grid --config experiment/configs/synthetic_grid.yaml
PYTHONPATH=experiment/src python3 -m eventclock.run_grid --config experiment/configs/synthetic_baseline_grid.yaml
PYTHONPATH=experiment/src python3 -m eventclock.run_grid --config experiment/configs/synthetic_token_baseline_grid.yaml
PYTHONPATH=experiment/src python3 -m eventclock.run_grid --config experiment/configs/synthetic_cnn_grid.yaml
```

Summarize runs:

```bash
PYTHONPATH=experiment/src python3 -m eventclock.summarize_results experiment/outputs --out experiment/outputs/summary.csv
```

Visualize a learned clock:

```bash
PYTHONPATH=experiment/src python3 -m eventclock.visualize_clock \
  --config experiment/configs/synthetic_eventclock.yaml \
  --checkpoint experiment/outputs/synthetic_eventclock/best.pt \
  --out experiment/outputs/synthetic_clock.png
```

## Preprocessed Dataset Format

All real datasets use the same NPZ schema so the training code does not depend on fragile raw-data download layouts:

```text
x: float32 array, shape [N, C, T] by default
y: int64 array, shape [N]
split: optional string array, values train/val/test
evidence_mask: optional float32 array, shape [N, T]
```

If `split` is absent, the loader uses deterministic 70/15/15 splits. For PTB-XL, Sleep-EDF, WESAD, and PPG-DaLiA, create the files referenced in:

- `experiment/configs/ptbxl_npz_eventclock.yaml`
- `experiment/configs/sleep_edf_npz_eventclock.yaml`
- `experiment/configs/wesad_npz_eventclock.yaml`
- `experiment/configs/ppg_dalia_npz_eventclock.yaml`

Validate a preprocessed file before training:

```bash
PYTHONPATH=experiment/src python3 -m eventclock.validate_npz data/processed/ptbxl_superdiag.npz
```

Then run:

```bash
experiment/scripts/run_dataset.sh experiment/configs/ptbxl_npz_eventclock.yaml
```

## Reviewer Checklist Mapping

- “Is this just adaptive patching?” Compare `event_clock` against `complexity_token` and tuned `fixed_patch`.
- “Does local density correspond to evidence?” Report deletion/insertion AUC and synthetic mask IoU.
- “Is the fixed patch baseline tuned?” Use `synthetic_baseline_grid.yaml` and patch-size sweeps on each real dataset.
- “Are gains robust?” Use `robustness_eval` blocks for noise, shift, masking, channel dropout, and amplitude scaling.
- “Is the effect stable?” Use `seed` grid values and summarize mean/std across runs.
- “Do evidence losses matter?” Use `loss.lambda_suff` and `loss.lambda_nec` ablations.
- “Does token budget matter?” Sweep `model.k_tokens` and plot performance vs K.

## Notes

The current repository does not include raw public datasets. The code intentionally fails with a clear `FileNotFoundError` if an NPZ path is missing, which keeps dataset licensing and storage separate from the experiment logic.
