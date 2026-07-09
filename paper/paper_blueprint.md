# Causal Event-Time Transformer: Paper Blueprint

## Working Title

**Causal Event-Time Transformer: Learning Evidence-Calibrated Clocks for Sparse Signals**

Alternative titles:

- **EventClock: Learning Adaptive Time for Sparse Signal Representation**
- **Evidence-Calibrated Event Clocks for Biomedical Time-Series Transformers**
- **Learning Task-Causal Time Reparameterizations for Sparse Sequential Signals**

## Core Thesis

Most time-series models operate on physical time or fixed patches. For sparse signals, physical time is a poor computational coordinate: a 10 ms arrhythmic ECG segment, a sleep-stage transition, or a motion artifact can carry more evidence than long stable periods.

The proposed method learns an **event-time coordinate** that expands task-relevant evidence and compresses redundant intervals under a fixed compute budget. Unlike ordinary adaptive patching, the clock is calibrated by counterfactual evidence objectives, so high clock density should correspond to regions whose removal changes the prediction.

## Main Research Question

Can a sequence model improve low-budget sparse-signal representation by learning a differentiable event-time coordinate whose density is calibrated to counterfactual task evidence?

## Contribution Claims

1. **Learned event-time reparameterization**: introduce a differentiable monotone map from physical time `t` to event time `tau(t)`.
2. **Budgeted clock constraint**: allocate a fixed total event-time budget across the sequence.
3. **Evidence calibration**: train the clock with sufficiency and necessity losses so selected intervals are not merely locally complex, but task-causal.
4. **Architecture plug-in**: use the event-time samples with Transformer, SSM, or CNN backbones.
5. **Cross-domain validation**: ECG, EEG/sleep, wearable stress, and optionally PPG/HAR.

## Method Sketch

Given signal `x(t)` for `t in [0, T]`:

```text
x(t) -> ScoutNet -> clock velocity v(t) >= 0
tau(t) = integral_0^t v(u) du
normalize tau(T) = B
sample K tokens from uniform event time grid in [0, B]
map samples back with tau^{-1}
encode selected windows with Transformer/SSM
predict y
```

The key architectural difference from dynamic patching is that patch boundaries are induced by a continuous latent clock, not directly chosen by a local complexity heuristic.

## Mathematical Formulation

Clock velocity:

```math
v_\theta(t) = \epsilon + \operatorname{softplus}(g_\theta(s_\theta(x, t)))
```

Event time:

```math
\tau_\theta(t) = B \cdot \frac{\int_0^t v_\theta(u)\,du}{\int_0^T v_\theta(u)\,du}
```

Event-time grid:

```math
r_k = \frac{k + 1/2}{K}B,\quad \hat{t}_k = \tau_\theta^{-1}(r_k)
```

Token extraction:

```math
z_k = \phi_\psi(x[\hat{t}_k - w_k/2,\hat{t}_k + w_k/2])
```

Prediction:

```math
\hat{y}=h_\omega(\operatorname{Backbone}(z_1,\ldots,z_K))
```

Base objective:

```math
\mathcal{L}_{base} =
\mathcal{L}_{task}(\hat{y}, y)
+ \lambda_r \mathcal{L}_{recon}
+ \lambda_s \int_0^T |\partial_t v_\theta(t)|\,dt
+ \lambda_e H(\bar{v}_\theta)
```

Evidence-calibrated objective:

```math
\mathcal{L}_{suff} =
D(f(x), f(x \odot m_\theta))
```

```math
\mathcal{L}_{nec} =
\max(0, \gamma - D(f(x), f(x \odot (1-m_\theta))))
```

where `m_theta` is a soft mask induced by high clock-density regions. The desired behavior is:

- keeping selected evidence preserves the prediction;
- removing selected evidence changes the prediction;
- removing unselected regions has smaller effect.

Total:

```math
\mathcal{L} =
\mathcal{L}_{task}
+ \lambda_{suff}\mathcal{L}_{suff}
+ \lambda_{nec}\mathcal{L}_{nec}
+ \lambda_{smooth}\mathcal{L}_{smooth}
+ \lambda_{budget}\mathcal{L}_{budget}
```

## Related Work Positioning

### Dynamic Patching

TimeSqueeze proposes content-aware dynamic patching for efficient time-series forecasting, using local signal complexity to allocate short patches to information-dense regions and long patches to redundant regions. This validates the fixed-patching problem but creates a close baseline that this paper must beat.

Source: https://arxiv.org/abs/2603.11352

### Adaptive Patching Critique

Adaptive patching is not automatically useful: local heterogeneity alone may not identify where finer patching reduces forecasting loss. This is the exact weakness EventClock should address via evidence calibration.

Source: https://arxiv.org/abs/2606.04074

### Continuous-Time Models

Neural CDEs model irregular time series in continuous time and show that sequence models can be formulated through controlled differential equations. EventClock is different: it learns a task-adaptive time coordinate for discrete downstream encoders rather than directly solving a continuous-time hidden state equation.

Source: https://arxiv.org/abs/2005.08926

### Shapelets and Rationales

Shapelets identify discriminative subsequences in time series; rationale and evidence bottleneck methods select input evidence. EventClock generalizes these ideas into a differentiable time-warped compute coordinate.

Shapelets source: https://dl.acm.org/doi/10.1145/1557019.1557122

## Paper Outline

1. **Introduction**
   - Sparse signals have non-uniform information density.
   - Fixed time and fixed patching waste compute and blur short evidence.
   - Existing adaptive patching often uses local complexity, not causal evidence.
   - Introduce evidence-calibrated event-time modeling.

2. **Problem Setup**
   - Sparse signal representation under token budget.
   - Define physical time, event time, budget, and evidence calibration.

3. **Method**
   - ScoutNet and clock velocity.
   - Event-time normalization and inverse sampling.
   - Event-token encoder.
   - Evidence sufficiency and necessity objectives.
   - Implementation details: differentiable interpolation, straight-through sampling option, smoothness/budget regularizers.

4. **Experiments**
   - Datasets: PTB-XL, Sleep-EDF, WESAD, PPG-DaLiA optional.
   - Tasks: classification, sleep staging, stress detection, HR estimation optional.
   - Baselines: fixed patch Transformer, tuned uniform patch, TimeSqueeze-style adaptive patch, attention pruning, random budget, saliency post-hoc selection.

5. **Results**
   - Accuracy/F1/AUROC under token budgets.
   - Deletion/insertion evidence curves.
   - Event alignment metrics.
   - Compute vs performance.
   - Cross-dataset robustness.

6. **Analysis**
   - Clock density visualizations.
   - Does `tau'(t)` align with R-peaks, sleep transitions, stress onsets?
   - Failure cases: noisy signals, label leakage, overly smooth clocks.
   - Ablations: remove necessity loss, remove sufficiency loss, remove smoothness, fixed vs learned budget.

7. **Conclusion**
   - Event-time as a general coordinate for sparse sequence modeling.
   - Limitations and extensions to scientific signals beyond biomedicine.

## Claims That Must Be Proven

- The method beats a carefully tuned uniform patch baseline, not just a weak fixed patch.
- The clock is not only a complexity detector.
- Counterfactual evidence losses improve event alignment and robustness.
- Gains persist across at least two signal domains.

## Reviewer Risk

Highest risk: reviewers call it "adaptive patching with a new name."

Defense:

- continuous monotone time reparameterization rather than patch selection;
- explicit counterfactual evidence calibration;
- evaluate deletion/insertion causal evidence, not only accuracy;
- include strong tuned-uniform and TimeSqueeze-style baselines.

