# Xavier, Kaiming, and What Modern LLMs Actually Use

The variance-preservation argument (file `01`) gives the right form for an init: `σ_W² = c / fan`. The constant `c` and the choice of `fan` depend on the activation function. This file is a tour of the specific schemes you'll see in code, plus what frontier-scale models actually use.

## Xavier / Glorot (2010) — for tanh / sigmoid

For activations with derivative ≈ 1 at the origin (tanh, sigmoid linearized around 0), no variance compensation is needed. To preserve variance in both directions (forward and backward), Glorot proposed splitting the difference between `fan_in` and `fan_out`:

```
σ_W² = 2 / (fan_in + fan_out)
```

This was the "Xavier" or "Glorot" init that powered the brief tanh-network revival of the early 2010s. Two variants:
- **Xavier normal**: `W ~ N(0, σ²)`.
- **Xavier uniform**: `W ~ Uniform(-√(6/(fan_in + fan_out)), +√(6/(fan_in + fan_out)))`. The `6` is `2 · 3` — `3` is the variance of `Uniform(-1, 1) · 3 = Uniform(-√3, √3)`.

Why the average? Because making forward perfect (`1/fan_in`) makes backward worse, and vice versa. The harmonic-mean compromise is the natural symmetric choice. For square layers it's exact.

## Kaiming / He (2015) — for ReLU

ReLU breaks the symmetry: it kills the negative half. As shown in file `01`, this halves the post-activation variance. He et al. compensated:

```
σ_W² = 2 / fan_in       (Kaiming-fan-in, the common variant)
```

Or `2 / fan_out` for the backward-optimized variant. Modern code defaults to `fan_in` for forward-pass-friendly init; the difference is marginal in practice when combined with residuals and normalization.

The "Kaiming" name attached to this in vision; in NLP code you'll see `kaiming_normal_` (PyTorch) or `he_normal` (TF/JAX). Same thing.

```python
torch.nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
```

This sets `σ² = 2 / fan_in`. For `nonlinearity='leaky_relu'`, the constant `2` is adjusted to account for the negative-side slope. For other activations (GELU, SiLU), the same `2/fan_in` is usually used as a heuristic; their negative-side suppression is similar enough that the empirical difference is in the noise.

## What modern LLMs actually use

The picture is messier than "Xavier or Kaiming." Most large open models use one of:

### 1. Plain Gaussian with a fixed small std

GPT-2, NanoGPT, many smaller training stacks:
```
W ~ N(0, 0.02²)
```

A constant `σ = 0.02` regardless of `fan_in` or `fan_out`. Note `0.02` is a **standard deviation**, not a variance — the distribution is `N(0, 0.02²)`, variance `0.0004`. Why does this work? Because for the typical Transformer widths (`D = 768` to `4096`), `√(2/D)` is between `0.022` and `0.051` — `0.02` sits in (just below) this range and the exact value isn't critical.

**Why landing slightly *too small* is the right error to make.** Notice `0.02` is a touch under the principled `√(2/D)`, and that's deliberate. In a pre-norm residual stack every block *adds* its output into the residual stream, so variance accumulates with depth — the stream grows as you go up the layers. Init each block's output projection at the full Kaiming scale and that growth compounds, so the early-training residual stream can blow up before LayerNorm has anything stable to renormalize against. Underscaling damps each block's contribution and keeps the stream calm at step 0. Erring large compounds across `L` layers; erring small just gets renormalized away by the next LayerNorm. (Recipe 2 below makes this explicit with the extra `1/√(2L)` damping on the residual-output projections — the constant `0.02` is the cheap version of the same instinct.)

**Why "small embedding dim" is the stated caveat.** The constant only stays in the Kaiming ballpark while `D` is in the hundreds-to-low-thousands. Push `D` to 8192+ and `√(2/D) ≈ 0.016` drifts *below* `0.02`, so the fixed constant becomes meaningfully too big and you're back to width-aware init (or careful residual scaling). `0.02` is a convenient constant that happens to land right for GPT-2-scale widths, not a width-independent law.

This is what you'll see in NanoGPT-flavored code. Not theoretically motivated, but works in practice.

### 2. Kaiming-style with `1/√(2L)` damping on output projections

LLaMA 2/3, Mistral, many modern open models:
- Most weights: `~ N(0, σ²)` with `σ² ≈ 1/fan_in` (variant) or `0.02` (constant).
- **Output projections of residual sublayers** (`W_O` in attention, `W_down` in FFN): scaled additionally by `1/√(2L)` where `L` is the number of layers.

This extra damping is the "init for depth" trick from file `01` — it keeps the residual stream's variance at O(1) at init even as depth grows.

```python
# GPT-2's recipe, in pseudo-code:
for module in model.modules():
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
for pn, p in model.named_parameters():
    if pn.endswith('c_proj.weight'):           # the residual-output projections
        nn.init.normal_(p, std=0.02 / math.sqrt(2 * n_layer))
```

### 3. μP (Maximal Update Parameterization)

Frontier-scale training stacks (Tensor Programs / μTransfer-style):
- Per-layer init scales depend on `width` in specific ways.
- LR is scaled per layer such that the effective parameter update is **width-independent**.
- Net effect: hyperparameters tuned at small width transfer exactly to large width.

This isn't a single formula but a set of co-designed init and LR scalings. The practical payoff: instead of doing a learning-rate sweep at 1T parameters (catastrophically expensive), you sweep at 1B parameters and apply the same numbers at 1T. Saves enormous compute. Used in some Anthropic and DeepMind training stacks.

The cost: a more complex codebase and a steeper learning curve. Not used in most open-source training (NanoGPT, axolotl, most fine-tuning) but increasingly common in serious pretraining work.

## Init for specific Transformer parts

A pre-norm Transformer has several distinct parameter groups, each with their own init conventions:

| Parameter | Typical init |
|---|---|
| Token embeddings `W_E` | `N(0, 0.02²)` — small, because embeddings are added to the residual stream and shouldn't dominate it at step 0. |
| Position embeddings (if learned) | `N(0, 0.02²)` — same logic. |
| Q, K, V projection weights | `N(0, σ²)` with `σ² ≈ 1/D` or `0.02`. Standard linear-layer init. |
| Output projection `W_O` (attention) | Same as above, optionally with `1/√(2L)` damping for residual-stream stability. |
| FFN gate / up projections | `N(0, 0.02²)` or Kaiming-style. |
| FFN down projection `W_down` | Same as `W_O`, often with the same `1/√(2L)` damping. |
| LN / RMSNorm `γ` (scale) | Constant `1` — start as identity. |
| LN `β` (shift) | Constant `0`. |
| Final output / unembedding `W_U` | Often **tied to `W_E`** (shared weight matrix). If not tied, initialized similarly. Sometimes zero-initialized so the model starts uniform over the vocab. |

## Tied embeddings — a common modern choice

`W_U = W_E^T` (the input embedding matrix is reused as the output unembedding). Two benefits:
- **Parameter savings**: a typical vocab is 32K–256K, `D` is 4096–16384. `V · D` is a lot of parameters — tying saves this much.
- **Inductive bias**: the same vector represents a token both as input and as output target, encouraging consistent geometry.

Not always used. Llama 1/2 untied. Llama 3 untied. PaLM untied. GPT-2 tied. T5 tied. It's a minor architectural choice with a small empirical effect that depends on vocab/D ratio.

## What about uniform vs. normal?

Doesn't matter as long as variance matches. `Uniform(-a, a)` has variance `a²/3`, so `a = √(3 σ²)` matches a Gaussian with variance `σ²`. PyTorch's `kaiming_uniform_` uses uniform with matched variance; `kaiming_normal_` uses Gaussian. Pick one and stop worrying about it.

## The "truncated normal" detail

PyTorch's default Linear init uses `kaiming_uniform_` (which is uniform). HuggingFace and some other stacks use **truncated normal** — Gaussian samples with values past ±2σ rejected and resampled. This avoids extreme outlier weights at init that could destabilize early steps. Differences are small but real for very deep models.

```python
nn.init.trunc_normal_(weight, std=0.02, a=-2*std, b=2*std)
```

## Self-check

1. Why does Kaiming use `fan_in` while Xavier uses the average of `fan_in` and `fan_out`? What goes wrong if you pick the "wrong" one for your activation function?
2. Why are token embeddings initialized to a small scale (`0.02`) instead of `1/√fan_in` like Linear layers?
3. The `1/√(2L)` damping is applied to specific weight matrices (`W_O`, `W_down`), not all of them. Why these specific matrices?

### Answers

1. Kaiming targets forward-pass variance preservation specifically, because ReLU networks were the first stacks where the *forward* signal vanishing was the dominant pathology — the empirical fix was forward-direction init. For a non-square layer (typical in attention: Q, K, V projections often have `fan_in ≠ fan_out` when using MHA→MQA→GQA collapses), pure forward-fan-in init gives bigger backward gradients than ideal, but residuals + clipping handle it. Xavier's average is more symmetric — better for tanh networks where saturation in either direction is bad. Picking the wrong one introduces a small constant factor off (1.5–2× variance off), which is fine for shallow stacks and a real problem only for deep stacks without residuals.
2. Token embeddings are *added* to the residual stream (and possibly to positional embeddings, then propagated through `L` layers). If each embedding had unit variance, the residual stream at step 0 would have variance 1 from the embedding alone, plus accumulated contributions from `L` layers — easily 10–100× too large. By the time it hits the first LN, that LN is trying to normalize against a much-larger-than-expected magnitude, which destabilizes early training. Initializing embeddings small (`σ = 0.02`) keeps the residual-stream variance manageable at step 0. The model then learns to scale embeddings up as needed.
3. `W_O` (attention output projection) and `W_down` (FFN output projection) are the *last* linear maps in their respective sublayers — their outputs are added to the residual stream. By damping these specifically, you control how much each sublayer contributes to residual-stream variance growth, without affecting the internal computation (Q/K/V projections, FFN gate/up projections). Damping `W_Q`, `W_K`, etc. would shrink the within-block computation itself, which is unnecessary and would slow learning. The output projection is the bottleneck — damp it and you control depth scaling. Damp the rest and you just slow training.

## Exercise

In PyTorch, manually re-init a small Transformer (e.g. 6 layers, `D = 384`) with three schemes:
1. Default (`nn.Linear`'s default = `kaiming_uniform_(fan_in)` with `a=√5`).
2. GPT-2 style: all weights `N(0, 0.02²)`, output projections damped by `1/√(2L)`.
3. Pure `N(0, 1)` (will probably explode).

Train each for a few hundred steps. Plot loss curves. The first two should converge similarly; the third should NaN out almost immediately. This makes "variance matters" tangible — you can break a model with bad init in <100 steps.
