# Related Work Notes: Causal Event-Time Transformer

## Primary Nearby Work

### TimeSqueeze: Dynamic Patching for Efficient Time Series Forecasting

Source: https://arxiv.org/abs/2603.11352

Relevance:

- Direct nearest neighbor.
- Motivates the weakness of fixed patching.
- Uses local signal complexity and variable patch sizes.

Gap for this project:

- EventClock should not be framed as another dynamic patcher.
- The contribution must be continuous latent time reparameterization plus evidence-calibrated objectives.
- Must compare against TimeSqueeze-style segmentation if implementation resources permit.

### Adaptive Patching Is Harder Than It Looks For Time-Series Forecasting

Source: https://arxiv.org/abs/2606.04074

Relevance:

- Important cautionary paper.
- Argues local heterogeneity alone does not guarantee dynamic patching improves forecasting loss.
- Requires tuned uniform patch baselines.

Implication:

- EventClock needs a stronger routing signal than local complexity.
- Counterfactual sufficiency/necessity is not optional; it is the main novelty shield.

### Neural Controlled Differential Equations for Irregular Time Series

Source: https://arxiv.org/abs/2005.08926

Relevance:

- Provides continuous-time sequence modeling foundation.
- Handles irregular sampling and partially observed multivariate time series.

Gap:

- Neural CDE uses continuous-time dynamics over observed controls.
- EventClock learns the computational time axis for tokenization/encoding.

### Time-Series Shapelets

Source: https://dl.acm.org/doi/10.1145/1557019.1557122

Relevance:

- Classical evidence-subsequence idea.
- Useful framing for interpretability.

Gap:

- Shapelets are discriminative subsequences, not a learned global monotone clock.
- EventClock can be viewed as differentiable dense shapelet allocation under budget.

## Datasets and Official Sources

### PTB-XL

Source: https://physionet.org/content/ptb-xl/

Notes:

- Large public 12-lead ECG dataset.
- PhysioNet describes 21,799 clinical 12-lead ECGs from 18,869 patients, 10 seconds long.
- Useful for ECG classification and clock alignment with QRS/ST/T-wave regions.

### Sleep-EDF Expanded

Source: https://physionet.org/content/sleep-edfx/1.0.0/

Notes:

- 197 whole-night polysomnographic recordings.
- Includes EEG, EOG, chin EMG, event markers, and hypnograms.
- Useful for sleep staging and transition/event-time analysis.

### WESAD

Source: https://archive.ics.uci.edu/dataset/465/wesad%2Bwearable%2Bstress%2Band%2Baffect%2Bdetection

Notes:

- Multimodal wearable stress dataset.
- 15 subjects.
- Includes BVP, ECG, EDA, EMG, respiration, body temperature, and acceleration.
- Useful for stress onset and multimodal event-time testing.

### PPG-DaLiA

Source: https://archive.ics.uci.edu/dataset/495/ppg%2Bdalia

Notes:

- 15-subject PPG and accelerometer dataset for heart-rate estimation in daily-life activities.
- Useful as optional third/fourth dataset, especially for motion artifacts.

## Baseline Families

1. **Fixed patch Transformer**
   - PatchTST-style fixed windows.
   - Must tune patch size to avoid unfair comparison.

2. **Point-token Transformer**
   - High resolution but expensive.
   - Useful as upper-bound under more compute.

3. **Dynamic/adaptive patching**
   - TimeSqueeze-style.
   - Local complexity segmentation.

4. **Attention/token pruning**
   - Select tokens after or inside the encoder.
   - Contrast with pre-encoding event-time allocation.

5. **Post-hoc saliency**
   - Integrated gradients / attention rollout.
   - Not a representation mechanism, but useful for evidence alignment comparison.

6. **Random or uniform budget**
   - Necessary negative control.

## Key Novelty Constraints

- Do not claim "adaptive patching" as the novelty.
- Do not rely only on local complexity.
- Do not use weak baselines.
- Do not present saliency-like heatmaps without deletion/insertion tests.

## Useful Keywords for Further Search

- dynamic patching time series
- adaptive tokenization time series Transformer
- differentiable time warping neural networks
- learned monotone time reparameterization
- counterfactual rationale learning
- sufficiency necessity rationale objective
- budgeted inference sparse sequence modeling

