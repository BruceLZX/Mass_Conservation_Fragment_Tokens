# Deep Research Preparation: Causal Event-Time Transformer

## Current Best Direction

The strongest version is not plain EventClock. The strongest version is:

> **Evidence-Calibrated EventClock**: learn a monotone event-time coordinate whose density is trained not only by task loss, but also by counterfactual sufficiency and necessity.

This directly addresses the main weakness identified in adaptive patching literature: local complexity does not necessarily equal task evidence.

## Refined Research Question

Can a differentiable event-time reparameterization, calibrated by counterfactual evidence objectives, improve sparse biomedical signal representation under fixed token budgets compared with fixed patching and local-complexity adaptive patching?

## FINER Assessment

| Criterion | Score | Justification |
|---|---:|---|
| Feasible | 4/5 | PTB-XL, Sleep-EDF, WESAD, and PPG-DaLiA are public. A minimal prototype can be implemented with PyTorch interpolation and standard backbones. |
| Interesting | 5/5 | It addresses a core sequence modeling problem: physical time is not the right compute coordinate for sparse evidence. |
| Novel | 4/5 | Dynamic patching is close, but evidence-calibrated continuous event-time is more structural. Novelty depends on strong formulation and baselines. |
| Ethical | 5/5 | Public de-identified datasets; no human-subject intervention. Need comply with dataset licenses and avoid clinical claims. |
| Relevant | 5/5 | Applies to ECG, EEG, wearable, and scientific signals; not limited to a single diagnosis task. |
| Average | 4.6/5 | Worth prototyping. |

## Deep Research Findings

### Finding 1: Fixed patching is a recognized bottleneck

TimeSqueeze argues that point-wise tokenization preserves temporal fidelity but is expensive, while fixed patching imposes uniform boundaries that may blur local dynamics. It proposes dynamic patching based on signal complexity.

Source: https://arxiv.org/abs/2603.11352

Implication:

- The problem is timely.
- The baseline is strong and must be included.

### Finding 2: Adaptive patching has a known failure mode

Adaptive patching can fail because local heterogeneity is not necessarily where finer patching reduces forecasting loss.

Source: https://arxiv.org/abs/2606.04074

Implication:

- The core routing signal should be task evidence, not local complexity.
- This supports the evidence-calibrated objective.

### Finding 3: Continuous-time modeling is accepted but solves a different problem

Neural CDEs model irregular time series directly in continuous time.

Source: https://arxiv.org/abs/2005.08926

Implication:

- EventClock should cite continuous-time models but distinguish itself as learning the compute coordinate.

### Finding 4: Public data supports multi-domain validation

PTB-XL, Sleep-EDF, WESAD, and PPG-DaLiA are sufficient for a credible multi-domain proof.

Sources:

- PTB-XL: https://physionet.org/content/ptb-xl/
- Sleep-EDF: https://physionet.org/content/sleep-edfx/1.0.0/
- WESAD: https://archive.ics.uci.edu/dataset/465/wesad%2Bwearable%2Bstress%2Band%2Baffect%2Bdetection
- PPG-DaLiA: https://archive.ics.uci.edu/dataset/495/ppg%2Bdalia

## Method Blueprint

### Input

```text
x in R^{C x T}
y: class label, sleep stage, stress label, or regression target
B: event-time budget
K: number of event tokens
```

### ScoutNet

Use a cheap temporal encoder:

- depthwise separable 1D CNN;
- small SSM;
- lightweight TCN.

Output:

```math
s_t = \operatorname{ScoutNet}(x)_{t}
```

### Clock Velocity

```math
v_t = \epsilon + \operatorname{softplus}(w^\top s_t + b)
```

Normalize:

```math
\tilde{v}_t = B \cdot \frac{v_t}{\sum_{j=1}^{T} v_j}
```

Discrete event time:

```math
\tau_t = \sum_{j \le t} \tilde{v}_j
```

### Event-Time Sampling

Uniform event grid:

```math
r_k = \frac{k + 1/2}{K}B
```

Inverse map:

```math
\hat{t}_k = \tau^{-1}(r_k)
```

In implementation:

- use differentiable linear interpolation over the cumulative clock;
- or use soft assignment from event grid points to physical time positions.

Soft assignment option:

```math
a_{k,t} =
\frac{\exp(-|\tau_t-r_k|/\sigma)}
{\sum_j \exp(-|\tau_j-r_k|/\sigma)}
```

Token:

```math
z_k = \sum_t a_{k,t}\phi(x_{t-w:t+w})
```

### Event Mask

Clock-derived soft evidence mask:

```math
m_t = \operatorname{sigmoid}((\tilde{v}_t - q_\alpha(\tilde{v}))/\eta)
```

where `q_alpha` is a percentile threshold.

### Losses

Task:

```math
\mathcal{L}_{task} = CE(h(z_{1:K}), y)
```

Sufficiency:

```math
\mathcal{L}_{suff} =
D(p_\theta(y|x), p_\theta(y|x \odot m))
```

Necessity:

```math
\mathcal{L}_{nec} =
\max(0, \gamma - D(p_\theta(y|x), p_\theta(y|x \odot (1-m))))
```

Budget:

```math
\mathcal{L}_{budget} =
\left(\sum_t \tilde{v}_t - B\right)^2
```

Smoothness:

```math
\mathcal{L}_{smooth} =
\sum_t |\tilde{v}_{t+1}-\tilde{v}_{t}|
```

Entropy/diversity optional:

```math
\mathcal{L}_{entropy} =
-\sum_t \bar{v}_t \log \bar{v}_t
```

Total:

```math
\mathcal{L} =
\mathcal{L}_{task}
+ \lambda_{suff}\mathcal{L}_{suff}
+ \lambda_{nec}\mathcal{L}_{nec}
+ \lambda_{smooth}\mathcal{L}_{smooth}
+ \lambda_{entropy}\mathcal{L}_{entropy}
```

## Experimental Plan

### Phase 0: Synthetic Sanity Check

Generate signals:

```text
background noise + sparse motif at random location -> label
```

Expected:

- learned clock density peaks at motif;
- removal of high-density region flips prediction;
- fixed patch under low budget fails more often.

### Phase 1: PTB-XL ECG

Task:

- superdiagnostic or diagnostic classification.

Inputs:

- start with single lead or 12 leads.
- 10-second ECG segments.

Expected event alignment:

- high clock density around QRS complexes and abnormal waveform regions.

Metrics:

- macro AUROC;
- F1;
- low-budget accuracy at K = 8, 16, 32, 64;
- deletion/insertion curves.

### Phase 2: Sleep-EDF

Task:

- epoch-level sleep stage classification.

Expected event alignment:

- transition boundaries;
- spindles/K-complex-like patterns if visible;
- high-density regions near stage changes.

Metrics:

- macro F1;
- Cohen kappa;
- stage-wise F1.

### Phase 3: WESAD

Task:

- stress vs non-stress classification.

Expected event alignment:

- stress onset;
- EDA/ECG/respiration transitions;
- motion artifacts if they affect labels.

## Baselines

Minimum required:

1. Fixed patch Transformer with tuned patch size.
2. Point-token CNN/Transformer where feasible.
3. Random K-token selection.
4. Local-complexity adaptive patching.
5. Attention pruning / top attention token selection.
6. EventClock without sufficiency/necessity.
7. EventClock without smoothness.

## Ablations

- remove sufficiency loss;
- remove necessity loss;
- vary budget `K`;
- vary smoothness weight;
- hard vs soft event sampling;
- local complexity clock vs task-trained clock;
- frozen clock vs jointly trained clock;
- single-domain vs cross-domain training.

## Key Evaluation Figures

1. Performance vs token budget.
2. Clock density over raw signal.
3. Warped event-time visualization.
4. Deletion/insertion curve.
5. Clock entropy/sparsity vs performance.
6. Failure examples.

## Smoke Test: 5-7 Days

Day 1:

- implement synthetic data and fixed patch baseline.

Day 2:

- implement ScoutNet, soft event-time assignment, and event-token encoder.

Day 3:

- add sufficiency/necessity losses and synthetic sanity check.

Day 4:

- load PTB-XL single lead or small subset.

Day 5:

- compare fixed patch vs EventClock at K = 16/32.

Day 6:

- produce clock plots and deletion/insertion curves.

Day 7:

- decide go/no-go.

Go criteria:

- synthetic motif localization works;
- PTB-XL low-budget performance beats tuned fixed patch or matches it with better evidence curves;
- deletion of high-clock regions causes larger performance drop than deletion of random or low-clock regions.

No-go criteria:

- clock collapses to uniform despite regularization;
- clock learns noise/artifact unrelated to task;
- tuned uniform patch consistently matches or beats all variants;
- evidence losses destabilize training without improving deletion/insertion.

## Implementation Notes

- Start with PyTorch only.
- Avoid complex inverse-CDF sampling initially; use soft assignment to event grid.
- Keep ScoutNet cheap so compute saving story remains credible.
- Cache preprocessed datasets.
- Log `v_t`, `tau_t`, masks, selected tokens, and prediction probabilities.

