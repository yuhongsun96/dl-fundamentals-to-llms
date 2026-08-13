# Part 3 — Residual Connections & Deep Networks

The bridge chapter. Part 2 introduced the residual connection as one of several tricks that make deep stacks trainable — mentioned in passing in the gradient-pathologies file and the norm-placement file. Part 3 promotes it to what it actually is: **the central organizing abstraction of the modern Transformer.** Everything in Parts 5–11 (attention writing to a shared stream, interpretability's "residual stream" view, why pre-norm won, why models can be pruned or layer-dropped) is a consequence of the one idea developed here.

This part is deliberately a **synthesis**, not a from-scratch build. The mechanics live in Part 2 and Part 1's supplementary — this part references them and spends its pages on the *why* and on the pieces those files deferred with "(Part 3)" forward-pointers.

## Structure

- **3.1 The Skip Connection** — the four load-bearing ideas:
  - `01` The degradation problem — why deeper *plain* nets got *worse*, and why the fix is a skip connection (with the Highway / LSTM-gating lineage that an NLP reader already half-knows).
  - `02` The gradient highway — the `I + ∂f/∂h` identity path, forward and backward, tied together as one picture.
  - `03` Ensemble of shallow paths — the "unraveled" view, effective depth, and why you can drop or reorder layers (stochastic depth, LayerDrop).
  - `04` The residual stream as abstraction — read / compute / add; superposition; the view that makes interpretability tractable.
- **3.2 Normalization Placement & Depth** — where the norm goes and how deep you can actually go:
  - `01` Normalization placement, revisited — pre- vs. post-norm as a *residual-stream* story, plus the at-scale stabilizers (DeepNorm, Sandwich-LN, QK-norm, logit soft-capping).
  - `02` Scaling the residual stream — the variance-grows-with-depth problem and its fixes (`1/√(2L)`, LayerScale, ReZero), then depth-vs-width and why frontier LLMs stop at ~tens-to-~100 layers instead of 1000.

## How to use

Read Part 2's [2.2/05](../part2_neural_network_fundamentals/2.2_backpropagation/05_gradient_pathologies.md), [2.3/03](../part2_neural_network_fundamentals/2.3_init_normalization/03_norm_layers.md), and [2.3/04](../part2_neural_network_fundamentals/2.3_init_normalization/04_pre_vs_post_norm.md) first — this part assumes them. If the residual connection already feels obvious to you, `01`/`02` will be fast; the payoff files are `03` (ensemble view) and `3.2/02` (scaling + depth-vs-width), which cover material not in Part 2.

## Target time

1–2 days. Shorter than Part 2 because it leans on it. The [residual stream supplementary](../part1_math_foundations/1.1_linear_algebra/supplementary/06_residual_stream.md) and [ARCHITECTURE.md](../ARCHITECTURE.md) are the two companion reads.

## What's deliberately omitted

- **The mechanics of specific norm layers** (BatchNorm/LayerNorm/RMSNorm formulas) — those are [2.3/03](../part2_neural_network_fundamentals/2.3_init_normalization/03_norm_layers.md). Here we only care about *placement* and *scaling*.
- **Vision ResNet specifics** (bottleneck blocks, downsampling shortcuts, the CIFAR/ImageNet curves) — we take the one idea that transferred to Transformers and leave the CNN detail behind.
- **Init math** — [2.3/01](../part2_neural_network_fundamentals/2.3_init_normalization/01_init_variance_preservation.md) and [2.3/02](../part2_neural_network_fundamentals/2.3_init_normalization/02_xavier_kaiming_modern.md) own that; `3.2/02` only uses the results.
