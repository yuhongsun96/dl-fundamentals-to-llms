# Inner Products, Norms, and Cosine Similarity

## Inner product

For `x, y ∈ R^d`:
```
<x, y> = x^T y = Σ x_i y_i
```

Geometrically: `<x, y> = ‖x‖ ‖y‖ cos(θ)`. The dot product measures **how aligned** two vectors are, scaled by their magnitudes.

This is the only operation in attention that actually compares tokens. Everything else is bookkeeping.

### Why it's called "inner"

The name tracks the shape of the underlying matmul:
```
x^T y :  (1, d) @ (d, 1) → (1, 1)        # scalar
```
The shared dim `d` sits **inside** — between the two outer `1`s — and gets summed away. Compare to the outer product (next file), where the shared dim is `1` and sits on the outside.

Rule of thumb: **inner collapses, outer expands.** The names are literally describing what the operation does to the shape.

### Geometric picture

```
      y
     ╱
    ╱
   ╱ θ
  ╱_________ x
      ⊢─proj─⊣
```

Drop a perpendicular from the tip of `y` onto the line through `x`. The signed length of that shadow is `‖y‖ cos θ`. Multiply by `‖x‖` to get `<x, y>`. Symmetric — you can drop `x` onto `y` and get the same number.

Sign interprets as:
- `> 0`: same general direction (angle < 90°)
- `= 0`: perpendicular (angle = 90°)
- `< 0`: opposite (angle > 90°)

**Mental model**: "how much of one vector lives along the other, times its length." A single number summarizing directional agreement between two vectors.

## Norms you need

A norm measures "how big" a vector is. Different norms answer "big in what sense?" differently, and the choice shapes how regularization, clipping, and optimization behave.

Quick reference:

| Norm | Formula | What it measures | Where you see it |
|------|---------|------------------|-----------------|
| L2 | `√(Σ x_i²)` | Euclidean length | Weight decay, gradient clipping, RMSNorm, cosine sim |
| L1 | `Σ |x_i|` | Sum of magnitudes | Sparsity regularizers |
| L∞ | `max |x_i|` | Largest component | Adversarial robustness, per-coord clipping |
| Frobenius | `√(Σ W_ij²)` | Matrix L2 | Weight regularization, low-rank approx error |
| Spectral | `σ_max(W)` | Max singular value | Lipschitz bounds, training stability |
| Nuclear | `Σ σ_i` | Sum of singular values | Low-rank-inducing penalty |

### What every norm has in common

Three properties: non-negative (zero only at origin); scales with multiplication (`‖α x‖ = |α| · ‖x‖`); triangle inequality (`‖x + y‖ ≤ ‖x‖ + ‖y‖`).

The **Lp family** parameterizes the most common norms by `p`:
```
‖x‖_p = (Σ |x_i|^p)^(1/p)
```
The shape of the **unit ball** (set of vectors with norm ≤ 1) is the visual key to each norm's behavior.

### L2 — the Euclidean norm

```
‖x‖_2 = √(Σ x_i²)
```

Straight-line distance from origin to `x`. The "ruler distance."

**Unit ball**: a sphere. Smooth, rotation-invariant, no preferred direction.

```
   ┌─────────┐
   │   ◯     │      L2 unit ball: circle / sphere
   │         │
   └─────────┘
```

**Why it dominates DL**:
- Smooth everywhere except the origin; differentiable. `∇‖x‖₂ = x / ‖x‖₂`.
- Rotation-invariant — matches the implicit Gaussian assumption that no direction is special.
- Matches the inner-product structure: `‖x‖₂² = <x, x>`. Squared L2 is what you actually compute and differentiate (avoids the square root).
- **Concentration of measure**: in high dim, almost all the mass of a Gaussian sits in a thin shell of L2 radius `√d`. This is why `√d` shows up everywhere (attention scaling, init schemes, embedding magnitudes).

**Where you see it**:
- **Weight decay**: add `λ ‖W‖₂²` to the loss → penalizes large weights → improves generalization. Almost universal.
- **Gradient clipping**: if global `‖∇‖₂ > threshold`, scale the entire flattened gradient down. The threshold is enforced on the L2 norm of all parameter gradients concatenated.
- **RMSNorm**: divides activations by their RMS = `‖x‖₂ / √d`. Pure per-sample L2 normalization with a learned scale.
- **Cosine similarity**: dot product divided by L2 norms.

### L1 — the Manhattan / taxicab norm

```
‖x‖_1 = Σ |x_i|
```

Sum of absolute values. "How far you'd walk on a city grid."

**Unit ball**: a diamond (octahedron in higher dim). Has sharp **corners on the coordinate axes**.

```
   ┌─────────┐
   │    ╱╲   │      L1 unit ball: diamond
   │   ╱  ╲  │      corners pointing along the axes
   │  ╱    ╲ │
   │  ╲    ╱ │
   │   ╲  ╱  │
   │    ╲╱   │
   └─────────┘
```

**The sparsity property**: if you constrain a vector to live inside an L1 ball while minimizing some loss, the optimum tends to land on a **corner** — meaning most coordinates are exactly zero.

- L2 ball is round → optimum can be anywhere on the boundary, usually with all coordinates nonzero.
- L1 ball has corners on the axes → optimum is *often* a corner → many coordinates exactly zero.

So L1 regularization (Lasso) induces sparsity; L2 (ridge) just shrinks weights uniformly. This is **geometric, not numerical**.

**Where you see it in DL**: less common than L2. Sparsity-inducing regularizers, L1 reconstruction losses (robust to outliers), some attention-sparsity work. Modern Transformers mostly stick with L2 because GPUs love smooth gradients and sparsity isn't usually the goal.

### L∞ — the max norm

```
‖x‖_∞ = max_i |x_i|
```

The largest absolute coordinate. Ignores everything except the most extreme entry.

**Unit ball**: a square (cube in higher dim), aligned with the axes.

```
   ┌─────────┐
   │ ┌─────┐ │      L∞ unit ball: square / cube
   │ │     │ │      bounded independently per coordinate
   │ │     │ │
   │ └─────┘ │
   └─────────┘
```

**Where you see it**:
- **Adversarial robustness**: an L∞ attack perturbs every pixel/feature independently up to `ε`. The dominant adversarial threat model in vision; defenses are evaluated against L∞-bounded attacks.
- **Per-coordinate gradient clipping**: bounds each parameter's gradient separately, instead of globally.
- **Quantization**: max activation magnitude determines the dynamic range you need to represent.

### Frobenius — L2 for matrices

```
‖W‖_F = √(Σᵢⱼ W_{ij}²) = √(trace(W^T W)) = √(Σ σᵢ²)
```

Three equivalent forms: sum of squared entries; trace of `W^T W`; sum of squared singular values. Just the L2 norm of the matrix flattened into one long vector — nothing matrix-specific about it.

**Where you see it**:
- **Weight regularization** for entire matrices: `λ ‖W‖_F²`.
- **Matrix approximation error**: "how close is the rank-`r` approximation to the original?" measured in Frobenius. Eckart-Young says SVD truncation is **optimal** in this norm.
- **LoRA convergence analyses**: "how big is the learned update?" reported in Frobenius.
- **Whole-model gradient norm**: in practice computed as a Frobenius / global L2 of the flattened parameter tree.

### Spectral and nuclear norms (matrix-specific)

These don't reduce to a vector norm of flattened entries — they care about the **singular values**.

**Spectral norm** `‖W‖_2 = σ_max(W)`: largest singular value. Bounds the maximum amplification:
```
‖W x‖_2 ≤ ‖W‖_2 · ‖x‖_2
```
Used in:
- Lipschitz-constrained models (e.g. WGAN's spectral normalization).
- Training-stability analyses — runaway spectral norm in some weight matrix often precedes a loss spike.
- Some attention-stability work (Q/K weight matrices with bounded spectral norm).

**Nuclear norm** `‖W‖_* = Σ σ_i`: sum of singular values. The "L1 norm of singular values" — induces **low-rank-ness** the way L1 induces sparsity. Used in:
- Low-rank matrix recovery and analyses.
- Some neural-net generalization theory (low nuclear norm ↔ "simple" function).

**The pattern**: applying L1, L2, or L∞ to the singular value vector gives you nuclear, Frobenius, and spectral norms respectively.

| Norm of singular values | Resulting matrix norm | Property induced |
|---|---|---|
| L1 | Nuclear | Low rank |
| L2 | Frobenius | Smooth shrinkage |
| L∞ | Spectral | Bounded amplification |

### How the vector norms compare

For any `x ∈ R^d`:
```
‖x‖_∞ ≤ ‖x‖_2 ≤ ‖x‖_1 ≤ √d · ‖x‖_2 ≤ d · ‖x‖_∞
```

**The bigger `p`, the more "max-like"; the smaller `p`, the more "sum-like."** The gap between L1 and L∞ can be a factor of `d` — huge in high dim. That's why "norm-bounded" perturbations look very different across norms: an L∞-bounded attack with `ε = 0.01` can move the L2 norm by up to `√d · 0.01`, which is large for image-sized inputs.

### Practical cheat sheet

| Goal | Norm |
|---|---|
| Penalize big weights smoothly | L2 (weight decay) |
| Force many weights to zero | L1 (sparsity) |
| Bound worst-case per-coord perturbation | L∞ |
| Measure "how much have weights moved" | Frobenius |
| Bound Lipschitz constant | Spectral |
| Encourage low-rank weight matrix | Nuclear |

### One-line summaries

- **L2** — Euclidean length; smooth; the default for everything.
- **L1** — sum of magnitudes; geometry produces sparsity.
- **L∞** — worst single coordinate; the adversarial-attack norm.
- **Frobenius** — L2 of a flattened matrix; matrix-level shrinkage and approximation error.
- **Spectral** — max singular value; Lipschitz and amplification bound.
- **Nuclear** — sum of singular values; sparsity in singular-value space → low rank.

## Cosine similarity

```
cos(x, y) = <x, y> / (‖x‖ ‖y‖)
```

A dot product with magnitudes normalized away. Range `[-1, 1]`. Used everywhere embeddings live: retrieval, CLIP's contrastive loss, semantic search, RAG.

**Why cosine and not raw dot product?** Because embedding magnitudes can drift during training and encode things you don't want (e.g. token frequency). Cosine isolates direction.

**Why dot product in attention then?** Because you *want* magnitudes to matter there — a token can be "confidently" attending or not. The √d scaling handles the variance.

## The √d scaling — where it comes from

If `q, k ∈ R^d` have **i.i.d.** components (independent and identically distributed — each drawn from the same distribution) with mean 0 and variance 1:
```
Var(<q, k>) = Σ Var(q_i k_i) = d
Std(<q, k>) = √d
```

Dividing by `√d` brings pre-softmax variance back to 1, keeping scores in a range where softmax stays responsive instead of collapsing to one-hot. That responsiveness is what allows gradients to flow back through attention — without it, softmax saturates and the whole layer becomes un-trainable.

The `√d` in attention is the most-cited instance of a broader rule: **at every step where information could grow or shrink, push it back to unit magnitude.** The same principle drives `1/√fan_in` init, RMSNorm/LayerNorm, and residual normalization — and it's needed even in middle layers where there's no immediate softmax.

For the full story (saturation mechanics, vanishing-gradient math through softmax, why scaling can't be deferred to the end of the network, and how the unifying principle ties together init, norm layers, and attention scaling): see [`supplementary/03_scaling_and_saturation.md`](supplementary/03_scaling_and_saturation.md).

## L2 normalization

`x̂ = x / ‖x‖` projects onto the unit sphere. Many modern techniques do this:
- CLIP normalizes both text and image embeddings before the contrastive loss.
- RMSNorm is essentially per-sample L2 normalization (scaled).
- QK-norm (some recent LLMs) normalizes Q and K before attention to stabilize training at scale.

## Self-check

1. If I scale all embeddings by 10×, does cosine similarity change? Does dot-product attention output change?
2. Why does gradient clipping use the L2 norm of the full flattened gradient rather than per-parameter norms?
3. What goes wrong if you remove the `/√d` in attention for a model with `d_head=128`?

### Answers

1. **Cosine sim**: unchanged — `cos(10x, 10y) = (100 <x,y>)/(10‖x‖ · 10‖y‖) = <x,y>/(‖x‖‖y‖)`. Magnitudes cancel; only direction matters. **Dot-product attention**: catastrophically changed — pre-softmax scores `<10q, 10k> = 100·<q,k>` are 100× larger. Softmax saturates to one-hot, gradients vanish. Even with `/√d`, you've effectively divided the temperature by 100 — attention crashes into argmax behavior.
2. The model is one connected computational graph; the natural unit of "update size" is the L2 norm of the full gradient vector. Per-parameter clipping would distort the relative magnitudes between layers and break the coordinated step direction. Global clipping preserves direction (just shrinks uniformly), keeping the optimization step well-conditioned during a loss spike. Practically: `clip_grad_norm_(model.parameters(), max_norm)` is the universal idiom.
3. Score std = `√128 ≈ 11.3`. Score gaps within a row routinely span 30+, triggering the 10¹³ saturation ratio. Softmax goes one-hot per row; Jacobian entries → 0 everywhere; gradients through attention vanish. Model can't train. With `/√d = /11.3`: scores have std 1, softmax stays soft, gradients flow.

## Exercise

Generate 10K random 768-dim vectors. Plot the distribution of pairwise dot products and pairwise cosine similarities. Now plot `dot / √768`. Convince yourself why the scaling matters.
