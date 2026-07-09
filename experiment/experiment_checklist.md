# Experiment Checklist: Causal Event-Time Transformer

## Folder Purpose

This folder is for implementation planning and experimental execution. The paper argument is in `../paper/`.

## Immediate TODO

- [x] Create repo scaffold: `src/`, `configs/`, `scripts/`, `notebooks/`, `outputs/`.
- [x] Implement synthetic sparse motif dataset.
- [x] Implement fixed patch baseline.
- [x] Implement soft EventClock sampling.
- [x] Add clock visualization.
- [x] Add sufficiency/necessity losses.
- [x] Add PTB-XL/Sleep-EDF/WESAD/PPG-DaLiA NPZ entry configs.
- [x] Add tuned fixed patch, random-token, complexity-token, and CNN baselines.
- [ ] Run PTB-XL small subset after preprocessed NPZ is available.
- [ ] Compare against tuned fixed patch after real-data runs complete.

## Minimal Model Components

```text
src/eventclock/
  data/
    synthetic_sparse.py
    external.py
    robustness.py
    models/
    event_clock.py
    baselines.py
    transformer.py
  losses.py
  metrics.py
  train.py
  evaluate.py
  visualize_clock.py
  run_grid.py
  summarize_results.py
```

## First Baselines

1. [x] CNN baseline.
2. [x] Fixed patch Transformer.
3. [x] Fixed patch Transformer with best patch size sweep.
4. [x] Random event-token selection.
5. [x] Complexity-token adaptive baseline.
6. [x] EventClock without evidence loss through ablation config.
7. [x] EventClock with evidence loss.

## First Metrics

- [x] task metric: AUROC/F1/accuracy depending on dataset;
- [x] budget metric: number of tokens K through config grid;
- [x] deletion AUC;
- [x] insertion AUC;
- [x] selected-region overlap with synthetic motif;
- [x] clock entropy regularizer;
- [x] epoch runtime.

## First Config Grid

```yaml
dataset: synthetic_sparse
K: [8, 16, 32]
clock_smoothness: [0.0, 0.01, 0.1]
lambda_suff: [0.0, 0.1, 1.0]
lambda_nec: [0.0, 0.1, 1.0]
backbone: [tiny_transformer]
seed: [0, 1, 2]
```

## Decision Gate

Continue only if EventClock shows at least one of:

- better low-budget performance than tuned fixed patch;
- similar performance with much better deletion/insertion evidence;
- clear synthetic motif localization;
- stable non-collapsed event clocks across seeds.
