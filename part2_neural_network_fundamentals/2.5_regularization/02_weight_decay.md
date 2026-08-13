# Weight Decay as Implicit Regularization

The one regularizer that survived the LLM revolution. Every modern LLM uses it. Mostly invisible (a single line of optimizer config), but it does real work and tuning it incorrectly damages training.

## What it does

At every optimizer step, shrink every weight matrix toward zero by a small fraction:
```
W ← (1 - η · λ) · W - η · (gradient update)
```

`λ` is the **weight decay coefficient**. Typical values for AdamW pretraining: `λ = 0.1`. For some setups (e.g. LoRA fine-tuning): `λ = 0.01` or `0`.

For full AdamW formulation: see file 2.4/01 — the "W" in AdamW is precisely about decoupling this shrinkage from the gradient update.

## Two equivalent framings

### Framing 1: L2 penalty (the textbook view)

Add `(λ/2) ‖W‖²` to the loss:
```
L_total = L_task + (λ/2) Σ_ij W_ij²
```

The gradient of this penalty is `λ W`. Plugging into vanilla SGD gives the shrinkage update above. This is "L2 regularization" or "ridge regularization."

### Framing 2: weight decay (the AdamW view)

Apply the shrinkage `(1 - η λ) W` directly, **outside** the gradient update. In AdamW this is the correct framing because adding `λ W` to the gradient and then running it through Adam's normalization would dilute the regularization for parameters with large gradient histories (see file 2.4/01).

For pure SGD, the two framings are equivalent. For any adaptive optimizer (Adam, RMSprop, Adafactor), they're meaningfully different. Always use the "weight decay" form with adaptive optimizers.

## Why it improves generalization

The mechanism is genuinely fuzzy — there's no clean theorem that says "weight decay → better test loss" without strong assumptions. But several intuitions:

1. **Counteracts random drift.** Without decay, parameter magnitudes drift upward during training as random updates accumulate. Decay provides a restoring force, keeping the parameter norm in a controlled range.
2. **Encourages smaller weights.** Smaller weights mean a function with lower Lipschitz constant (less sensitive to small input changes), which usually generalizes better.
3. **Implicit prior on parameters.** From a Bayesian perspective, weight decay is a Gaussian prior on `W` centered at 0. Mild prior, but better than the uniform "any weight is equally likely" implicit prior.
4. **Effective parameter reduction.** Weights pushed close to zero contribute little to the function's expressive capacity. Decay effectively reduces the number of "active" parameters, an Occam's razor.

In practice: removing weight decay from LLM pretraining demonstrably hurts final eval loss. Adding it back recovers the loss. It's doing something useful; the precise mechanism is less important than the empirical regularity.

## Don't decay everything

The standard rule:
- **Decay**: 2D weight matrices (linear layers, embeddings if you're sloppy — see below).
- **Don't decay**: bias vectors, normalization parameters (`γ`, `β`), sometimes embeddings.

The standard PyTorch idiom:
```python
decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
optimizer = AdamW([
    {'params': decay_params, 'weight_decay': 0.1},
    {'params': nodecay_params, 'weight_decay': 0.0},
])
```

**Why exclude biases**: a constant offset doesn't add to the function's complexity in a meaningful way. Decaying biases pushes them toward 0, which is rarely what the model needs. Cost: small. Benefit: usually zero. Standard to exclude.

**Why exclude norm parameters**: `γ = 1` and `β = 0` are sensible defaults; decay would push them toward 0, which for `γ` is bad (zeroing out the scale collapses signal). For `β` it's fine (already initialized to 0). Just easier to exclude both.

**Embeddings — depends on convention**:
- GPT-2 / NanoGPT: decay embeddings. They're 2D, so the `p.dim() >= 2` filter includes them.
- Some Llama-style codebases: don't decay embeddings. Reasoning: embeddings are inherently sparse (only updated for tokens that appeared in the batch), and decay applied at every step on every embedding is disproportionate.
- Either is defensible. The empirical difference is small.

If your code has a "weight decay 0.1, decay everything 2D" rule, you're decaying embeddings. If it has special handling to exclude embeddings, fine.

## Tuning weight decay

`λ = 0.1` is the modern AdamW LLM default. The reasoning:
- AdamW's shrinkage per step is `η · λ`. With `η = 3e-4` and `λ = 0.1`, that's `3e-5` per step.
- Over 100K training steps, this compounds to a multiplicative `(1 - 3e-5)^100000 ≈ 0.05` — a 95% shrinkage if the gradient never countered it.
- The gradient does counter it, of course. The equilibrium where gradient and decay balance gives roughly stationary parameter norms.

Higher `λ` → more shrinkage → more regularization → less overfitting / lower effective capacity.
Lower `λ` → less shrinkage → more capacity, more overfitting risk.

For pretraining: `λ = 0.1` is the standard.
For fine-tuning: smaller, `λ = 0.01` to `0.05`.
For LoRA: often `λ = 0` (the LoRA matrices are already small).

## A subtle interaction with learning rate

Because AdamW's shrinkage is `η · λ`, doubling the learning rate also doubles the per-step shrinkage. If you change LR you should think about whether weight decay needs adjustment.

Some recent work (Loshchilov-Hutter follow-ups) argues for "decoupled WD" in a stricter sense: shrinkage is `λ · (1 - η)` or `λ / batch_size` — fully independent of LR. This is what frameworks like Lion implement. For AdamW specifically, the coupling is a minor wart but standard.

## Weight decay vs. L2 (one more time)

Different optimizers, different stories:

| Optimizer | L2-as-loss-term | AdamW-style decoupled |
|---|---|---|
| SGD | Equivalent | Same |
| SGD + momentum | Slightly different (decay enters momentum buffer) | Cleaner |
| Adam | **Materially different** — L2 is normalized by `√v_t` | **Use this** |
| Lion / others | Varies | Check the paper |

If you ever read "weight decay = L2 regularization," that's only true for plain SGD. For everything else they're different. AdamW exists to be explicit about which one you're doing.

## What weight decay isn't

- **Not a memorization preventer**. Doesn't directly stop the model from memorizing specific training examples; it's a magnitude regularizer.
- **Not a sparsifier**. L2 / weight decay shrinks weights smoothly — none go to exactly zero. L1 regularization is what induces sparsity (file 1.1/03).
- **Not a substitute for data**. Weight decay helps when you have enough data to train a useful model but want better generalization. With too little data, decay alone won't save you.

## Self-check

1. AdamW decouples weight decay from the gradient update. Why does this matter for Adam specifically but not for SGD? (Answered in 2.4/01, but worth re-deriving.)
2. The standard rule excludes bias and norm parameters from decay. Why is decaying a bias "bad" but decaying a weight matrix "good"?
3. With `λ = 0.1`, `η = 3e-4`, the per-step shrinkage is `3e-5`. Over 100K steps, the multiplicative shrinkage in absence of gradient is `0.05` (95% reduction). Why doesn't the model's weights actually collapse to 5% of their starting value?

### Answers

1. In SGD, the update is `θ ← θ - η · g`. Adding L2 makes `g = g_task + λ θ`, so `θ ← θ - η · g_task - η · λ · θ = (1 - ηλ) θ - η g_task`. Identical to "shrink then update." In Adam, the update is `θ ← θ - η · m / √v`. Adding L2 makes `g = g_task + λ θ` enter `m` and `v` — the `λ θ` term is averaged and squared-normalized. The effective decay applied to coordinate `i` is `λ · m_θ_i / √v_i`, which depends on per-coordinate gradient history. Coordinates with large historical gradients have large `√v_i` → small effective decay; coordinates with small historical gradients get amplified decay. The "shrinkage strength" becomes per-parameter and uncontrolled. AdamW removes this by applying shrinkage directly: `θ ← (1 - ηλ) θ - η · m / √v`. Now every parameter has the same fractional shrinkage `ηλ` per step. Predictable, simple, and empirically better.
2. The role of a bias is to provide a constant offset: `y = Wx + b` lets the layer produce a non-zero output for zero input, or shift the activation kink. Shrinking `b` toward 0 removes this offset capacity. If the optimal `b` is, say, `1.5`, decay constantly fights against it; gradient fights back, an equilibrium forms but it's lower than `1.5`. The model loses some expressivity in exchange for a regularization benefit that doesn't really apply (biases don't drive overfitting). Weight matrices, by contrast, contribute to the function's "wiggle" — decaying them genuinely controls complexity. Empirically: decaying biases hurts a little, decaying matrices helps a little. So: don't decay biases. (Same logic for `γ` in norm layers: shrinking `γ` toward 0 collapses signal.)
3. The gradient pushes back. Every step, weights receive both a shrinkage (`-η λ W`) and a gradient update (`-η m_t / √v_t`). If the gradient pushes weights *toward* their current value (i.e. the loss is locally minimized), shrinkage wins and weights move toward 0. If the gradient pushes them *away* from 0 (which it does for any feature the model is actively trying to learn), the gradient wins and weights stay nonzero. The equilibrium is roughly where these forces balance — weight norms stabilize at some training-dependent value rather than collapsing. You can verify this by logging weight norms during training: with decay, they grow during warmup, then stabilize. Without decay, they keep drifting upward indefinitely.

## Exercise

Train a small Transformer on a moderate LM task twice:
1. AdamW with `λ = 0.1`.
2. AdamW with `λ = 0`.

Track: (a) train loss, (b) validation loss, (c) the L2 norm of each weight matrix throughout training.

In (1), weight norms should stabilize after warmup. In (2), they should slowly grow throughout training. Validation loss should be slightly worse in (2). This makes weight decay's effect visible at every level — magnitude control, generalization, training dynamics — without any one experiment doing all the work.
