# Supplementary: `√d` scaling, softmax saturation, and why scaling matters at every layer

Companion to `../03_inner_products_norms.md`. Two related deep dives:

1. The full story behind the `√d` factor in attention — what saturation means, why it kills gradients, and why scaling fixes it.
2. The same scaling principle applied across the whole network — why `1/√fan_in` init, RMSNorm, residual scaling, etc. all matter at every layer, not just before softmax.

Read this if you want the *why* beyond "it keeps the variance unit." Otherwise the one-line version in the primary file is enough.

## Part 1 — The `√d` story in attention

### Variance of an unscaled dot product

If `q, k ∈ R^d` have components that are **i.i.d.** (independent and identically distributed — each component drawn independently from the same distribution) with mean 0 and variance 1, then:
```
Var(<q, k>) = Σ Var(q_i k_i) = d
Std(<q, k>) = √d
```

Without scaling, dot products grow with `d`, pushing softmax into **saturation** — and the consequences cascade.

### What saturation actually means

When pre-softmax scores are moderate, softmax produces a soft distribution like `(0.6, 0.3, 0.1)` that responds smoothly to score changes. When scores get extreme, softmax collapses to nearly one-hot — `(0.999, 0.0001, ...)`. That collapsed regime is saturation.

The exponential is unforgiving:

| score gap | probability ratio |
|---|---|
| 0 | 1× |
| 5 | ~150× |
| 10 | ~22 000× |
| 20 | ~5 × 10⁸× |
| 30 | ~10¹³× |

A 30-unit gap → 10-trillion-fold probability gap. Whichever score is biggest wins, hard.

### Why bigger `d` makes saturation worse

Without scaling, scores have std `√d`. For `d = 64` (typical head dim), std ≈ 8 — so scores within an attention row routinely span `[-20, +20]`, gaps of 30+ between max and median. The 10¹³ ratio kicks in. Softmax is forced into one-hot territory.

For `d = 1`, std ≈ 1, scores stay in `[-3, +3]`, softmax produces gentle distributions. Bigger `d` → larger pre-softmax range → more saturation, unless you scale.

### The actual killer: vanishing gradients through saturated softmax

The softmax Jacobian is:
```
∂s_i / ∂x_j = s_i (δ_ij - s_j)
```

In the saturated regime where `s_winner ≈ 1` and `s_other ≈ 0`:
- `∂s_winner / ∂x_winner = s_winner (1 - s_winner) ≈ 1 · 0 = 0`
- `∂s_winner / ∂x_other = -s_winner · s_other ≈ -1 · 0 = 0`
- Every entry of the Jacobian is ~0.

Gradients flowing back through softmax get multiplied by these ~0 entries → backward signal dies → **the model can't learn out of a wrong but confident prediction**. Forward produces a confident mistake; backward produces no signal to correct it.

This is what "vanishing gradient" means here — not that the loss has no gradient, but that the chain-rule path through saturated softmax kills it before it reaches the weights upstream.

### How `/√d` fixes it

Dividing by `√d` brings the pre-softmax variance from `d` back to 1:
```
Var(<q, k> / √d) = d / d = 1
```

Scores now sit in `[-3, +3]` regardless of head dim. Softmax stays in its soft, responsive regime. Jacobian entries are O(1). Gradients flow.

The `√d` isn't a numerical nicety — it's what makes attention trainable at all for non-trivial head dimensions.

### The asymmetry: scaling down is recoverable, scaling up isn't

A natural follow-up: what if real `Q` and `K` after init or training don't have exactly unit-variance i.i.d. entries? Then dividing by `√d` might over- or under-correct. Does that break things?

The answer hinges on an asymmetry between the two failure modes:

- **Scores trend small** (variance `<< 1`): softmax becomes nearly uniform — `(0.3, 0.3, 0.2, 0.2)` instead of `(0.6, 0.3, 0.1)`. Attention is unfocused, but **gradients still flow**. Jacobian entries `s_i(δ_ij - s_j)` stay nonzero. Training can recover and grow Q/K magnitudes if sharper attention helps the loss.
- **Scores blow up** (variance `>> 1`): softmax saturates to one-hot. Jacobian → 0. **No gradient flows.** Training is dead.

The whole point of `/√d` is to err on the safe side. You'd rather scores be a bit too small (recoverable, the model can learn out of it) than a bit too large (catastrophic). `√d` is the conservative choice that ensures even at maximum init variance you're not in saturation territory; if actual variance turns out smaller, that's fine.

### `/√d` is per-block, not compounded across layers

Another reasonable worry: "if `/√d` shrinks scores at every layer, won't they compound to nothing across 100 layers?"

No — `/√d` is applied **once per attention block** to the QK product `Q K^T / √d`. It doesn't iterate across layers. Each layer has its own attention with its own `/√d`. There's no multiplicative chain `(1/√d)^L` accumulating.

The thing that *would* compound across layers — generic activation magnitude drift — is killed by two other mechanisms:

- **Residual connections** add the unmodified input back: `x_{l+1} = x_l + f_l(x_l)`. Even if attention output trends to zero, the residual stream carries the signal forward at full scale. Worst case, a layer's attention becomes noise on top of the residual; the residual itself isn't shrunk.
- **Pre-norm RMSNorm/LayerNorm** rescales the input to each sub-block to unit RMS: `x_{l+1} = x_l + f_l(LayerNorm(x_l))`. Whatever magnitude `x_l` has drifted to, `f_l` always sees a unit-scale input. Drift can't compound because every layer renormalizes.

So inter-layer scale stability is a property of residuals + norm layers, not of `/√d`. The `/√d` only handles the local concern of "don't let a single softmax saturate."

### What actually happens during training

At init with proper Xavier-style scaling, `Q` and `K` have roughly unit-variance i.i.d. entries, and `/√d` does its job. As training progresses:

- The model learns Q and K with whatever magnitude the loss prefers.
- If softer attention works better (e.g., attending broadly), gradients shrink Q/K magnitudes; scores trend small; softmax becomes more uniform.
- If sharper attention works better (e.g., induction heads, BOS-attending heads), gradients grow Q/K; scores get larger; softmax becomes more peaked.
- The `/√d` is just a fixed offset; the model self-adjusts on top of it.

Empirically, after training, attention scores in different heads sit at varied stats. Some heads are deliberately sharp (intentional saturation, with low gradient through that head — but the rest of the model still trains via other paths); most are in a moderate-to-soft regime.

### QK-norm — the modern addendum for very large models

Recent observation (Henry et al. 2020 "Query-Key Normalization"; deployed in PaLM-2, Gemma, some Llama variants and 2024+ frontier stacks): at very large scale, even with `/√d`, **QK magnitudes can drift unboundedly during long training runs**. Some heads' Q/K weights grow without bound, scores explode mid-training, and you get loss spikes.

**QK-norm** L2-normalizes Q and K per head before the dot product:
```
score = (Q / ‖Q‖) · (K / ‖K‖)^T · γ
```

This pins scores into `[-1, +1]` regardless of weight magnitudes, with a learnable scalar `γ` (per-head learnable temperature) that the optimizer controls explicitly. Belt-and-suspenders on top of `/√d`: the `/√d` handles init, QK-norm handles the long-run drift.

Not universally adopted yet, but increasingly common in very large models where mid-training instability is expensive.

## Part 2 — Why this matters at every layer, not just attention

A natural question: in middle layers there's no immediate softmax. Why does scaling (`1/√fan_in` init, RMSNorm, residual normalization) still matter? Couldn't you defer all the rescaling to the end?

The short answer: **deferring works only in a purely linear network with infinite precision.** Nonlinearities, finite floating-point precision, residual streams, and gradient flow all force you to control magnitudes at every step.

Four reasons, each independently sufficient.

### Reason 1: Nonlinearities are not scale-equivariant

Softmax is the most dramatic case, but every nonlinearity has a "responsive regime" of input magnitudes outside which it stops doing anything useful.

| Nonlinearity | Saturation behavior |
|---|---|
| Softmax | Beyond `±5` per-row, mass collapses to one entry; Jacobian → 0 |
| Sigmoid / tanh | Beyond `±5`, output flattens to ±1, derivative → 0 |
| GELU / SiLU | Beyond `±10`, becomes effectively linear (curvature gone — no nonlinearity contribution) |
| ReLU | Scale-equivariant on the positive side, but if pre-activations are huge, the negative-vs-positive split is determined entirely by initialization noise — gating becomes useless |

If you defer all scaling to the end, every intermediate nonlinearity sees out-of-regime inputs and either saturates (softmax/sigmoid/tanh — gradients die) or degenerates to linear (GELU/SiLU — the network loses representational depth). **You can't compose a deep nonlinear network out of layers whose nonlinearities are all in the wrong regime and recover by rescaling the output.** The lost information is gone.

### Reason 2: Finite precision can't carry the extremes

Suppose you do 30 matmuls without proper init, with `W ~ N(0, 1)` instead of `N(0, 1/d)`. Each matmul scales activation variance by `d`. After 30 layers with `d = 4096`:
```
Var = d^30 = 4096^30 ≈ 10^108
```

Even bf16 (`max ≈ 10^38`) overflows to `inf` after a handful of layers. fp16 (`max ≈ 65k`) overflows in the first layer. Once you have `inf` anywhere, it propagates and the gradient becomes `nan`. Training is dead.

You can't "fix this at the end" because there's no end to reach — the forward pass has already corrupted itself with `inf`/`nan` values. No downstream scaling can resurrect that information. **Even in a purely linear network**, the intermediate values must be representable.

### Reason 3: The residual stream accumulates

Transformers add layer outputs to a shared residual stream:
```
x_{l+1} = x_l + f_l(x_l)
```

If each `f_l` outputs activations with variance `σ²`, after `L` independent additions the residual stream has variance `L · σ²` — std grows by `√L`. For a 100-layer model, that's a 10× growth just from the residual path.

Without LayerNorm/RMSNorm at each block to renormalize, the inputs to layer 100 are ~10× bigger than layer 1's inputs — pushing every nonlinearity (softmax included) into saturation in the deeper layers. The model trains the early layers fine, then the late layers stop contributing. This is exactly the failure mode pre-norm Transformers solve: ensure every layer's input is unit-scaled regardless of how much accumulated upstream.

### Reason 4: Gradient flow needs each layer's Jacobian to be O(1)

Backprop multiplies Jacobians across layers. If each layer's Jacobian has spectral norm `α`, then after `L` layers the gradient magnitude scales by `α^L`:

- `α > 1`: exploding gradients
- `α < 1`: vanishing gradients
- `α ≈ 1`: stable

Proper init (`W ~ N(0, 1/fan_in)`) is precisely the choice that makes each layer's Jacobian have spectral norm ~1 in expectation. Defer the scaling and `α ≈ √d` per layer; for `L = 100, d = 4096`, gradients scale by `√4096^100 ≈ 10^180`. They explode through the upstream layers and the optimizer can't take a sensible step.

This is a *gradient* problem, not just an activation problem. Even if forward magnitudes were somehow representable, backward Jacobians compound multiplicatively. Per-layer scaling controls forward magnitude **and** backward Jacobian magnitude in one move.

### Why "defer to the end" can't work — summary

It works *mathematically* in a purely linear network with infinite precision: a chain of linear maps is one big matrix, rescale once. Three things break that:

1. **Nonlinearities** care about local magnitudes; can't be undone by downstream scaling.
2. **Finite precision** — intermediate values overflow before reaching the end.
3. **Gradients** — backward Jacobians need to be O(1) per layer, not just at the output.

### The unifying picture

`1/√d` and its cousins (`1/√fan_in` init, RMSNorm's implicit `√d`, attention's `1/√d`, residual scaling) are all instances of the same principle:

> **At every step where information has the potential to grow or shrink, push it back to unit magnitude.**

This isn't normalization for normalization's sake. It's the only way to keep a deep network simultaneously:

- (a) using its nonlinearities (preventing saturation)
- (b) numerically representable (preventing overflow/underflow)
- (c) gradient-trainable (preventing exploding/vanishing gradients across depth)

The Transformer's `1/√d` in attention is the most-cited example because it's specifically about softmax saturation. View it as one application of a network-wide invariant: **maintain unit scale at every layer**. Init schemes do it for matmuls, normalization layers do it for activations, attention scaling does it for the QK product. Take any of them away and the network either doesn't train or trains to garbage.
