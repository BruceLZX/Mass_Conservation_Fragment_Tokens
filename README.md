# Causal Event-Time Transformer

Research scaffold for **Evidence-Calibrated EventClock**, a differentiable event-time tokenizer for sparse biomedical and wearable time-series signals.

The current implementation lives under [`experiment/`](experiment/):

- synthetic sparse-motif sanity checks;
- EventClock Transformer model;
- CNN, fixed-patch, random-token, and local-complexity baselines;
- sufficiency/necessity evidence losses;
- deletion/insertion evidence metrics;
- robustness and token-budget sweeps;
- NPZ entry points for PTB-XL, Sleep-EDF, WESAD, and PPG-DaLiA.

Quick smoke test after installing dependencies:

```bash
python3 -m pip install -r experiment/requirements.txt
experiment/scripts/run_synthetic_smoke.sh
```

See [`experiment/README.md`](experiment/README.md) for full experiment commands and dataset format.

