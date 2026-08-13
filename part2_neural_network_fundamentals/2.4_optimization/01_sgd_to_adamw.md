# SGD → Momentum → Adam → AdamW

The optimizer is the rule by which gradient information becomes a parameter update. Four generations:

- **SGD** — use the gradient directly.
- **Momentum** — average gradient over recent steps.
- **Adam** — also normalize by per-parameter recent magnitude.
- **AdamW** — fix Adam's interaction with weight decay.

Every modern LLM trains with AdamW or a close cousin. Understanding the four-step evolution is the cleanest way to see *why*.

## SGD

The base case. Given gradient `g_t = ∇L(θ_t)`:
```
θ_{t+1} = θ_t - η · g_t
```

`η` is the learning rate. That's it.

**Strengths**: simple, low memory (no extra state), well-understood theoretically.
**Weaknesses**: noisy (one minibatch is a noisy estimate of the true gradient), step direction can zigzag in ravines, requires careful LR tuning. Slow convergence on objectives with poor conditioning (most of deep learning).

SGD trains LLMs poorly. Not for any deep reason — just empirically. It needs much more wall-clock and is less stable. Modern training uses an Adam variant.

## SGD + Momentum

Add a running average of past gradients:
```
v_t = β · v_{t-1} + g_t
θ_{t+1} = θ_t - η · v_t
```

Symbol by symbol:
- `θ_t` — the parameters (weights) at step `t`; what we update.
- `g_t` — the gradient `∇L(θ_t)` on the current batch; points uphill, so we subtract it.
- `v_t` — the **velocity**: a running, decaying accumulation of past gradients (the momentum buffer / extra state).
- `β` — the momentum coefficient (`0.9` canonical); fraction of the old velocity carried forward, giving an effective memory of `~1/(1-β) ≈ 10` recent steps.
- `η` — the learning rate (step size).

Line 1 says "new velocity = most of the old velocity (`β · v_{t-1}`) + this step's gradient (`g_t`)". Line 2 is the SGD step, but stepping along the smoothed `v_t` instead of the raw `g_t`. `v_t` is an exponentially-decayed sum of past gradients — unrolled, `v_t = g_t + β·g_{t-1} + β²·g_{t-2} + ...` — so the update direction is smoothed and noisy components average out.

One scale subtlety: here `g_t` enters with coefficient `1` (not `1-β`), so for roughly steady gradients the velocity saturates near `g/(1-β) ≈ 10g` — momentum also amplifies the effective step by `1/(1-β)`, and `η` is often lowered to compensate. The true-EMA form `v_t = β·v_{t-1} + (1-β)·g_t` keeps the velocity at the gradient's scale; that's the convention Adam's first moment `m_t` below uses.

Intuition: imagine a ball rolling down a hill. SGD is friction-dominated (responds instantly to local slope); momentum gives the ball inertia (averages over recent slope history). It crosses small bumps that would stop SGD and accelerates along consistent gradient directions.

A small notation note: there's a variant ("Nesterov momentum") that evaluates the gradient at the **look-ahead** point `θ_t - η · v_{t-1}` instead of at `θ_t`. Marginal practical improvement in DL; mostly relevant for theory papers. PyTorch SGD with `nesterov=True` toggles this.

## Adam (Kingma & Ba, 2014)

The leap. Track two moving averages: not just the gradient (`m_t`, the first moment) but also the squared gradient (`v_t`, the second moment):
```
m_t = β1 · m_{t-1} + (1 - β1) · g_t                  # mean of recent gradients
v_t = β2 · v_{t-1} + (1 - β2) · g_t²                 # mean of recent squared gradients (per-coord)

m̂_t = m_t / (1 - β1^t)                                # bias correction
v̂_t = v_t / (1 - β2^t)

θ_{t+1} = θ_t - η · m̂_t / (√v̂_t + ε)
```

Defaults: `β1 = 0.9`, `β2 = 0.999`, `ε = 1e-8`.

Symbol by symbol:
- `m_t` — **first moment**: an EMA of the gradient. This *is* the momentum/velocity from above, in true-EMA form (the `(1-β1)` factor).
- `v_t` — **second moment**: an EMA of the *squared* gradient `g_t²` (elementwise), kept **per coordinate**. Always positive; tracks gradient *size*, not direction.
- `β1` — first-moment decay (`0.9`); ~10-step memory. `β2` — second-moment decay (`0.999`); ~1000-step memory, deliberately longer since a magnitude estimate should be smooth.
- `m̂_t`, `v̂_t` — the bias-corrected moments. `t` — the step number (the exponent in `β^t`).
- `η` — learning rate. `ε` — a tiny constant (`~1e-8`) to prevent division by zero.

Line by line:
- **Lines 1–2** track two running averages with the same EMA recipe: one on `g` (*which way are we heading?* — momentum) and one on `g²` (*how big is this coordinate's gradient, lately?* — magnitude, per weight).
- **Lines 3–4 (bias correction)** fix the fact that `m_t, v_t` start at 0 and are therefore biased toward zero early on. At `t=1`, `m_1 = (1-β1)·g_1 = 0.1·g_1` — ten times too small; dividing by `(1 - β1^1) = 0.1` multiplies by 10 and **exactly cancels** the initial shrinkage. As `t` grows, `β^t → 0` so the divisor `→ 1` and the correction switches off. Without it, Adam takes tiny, wrong-sized steps at the start (and the slower-warming `v̂` matters for ~the first thousand steps).
- **Line 5** steps in the smoothed direction `m̂_t`, but divides each coordinate by `√v̂_t` (its own typical magnitude); `+ ε` guards the division.

The key intuition for the second moment: **`v_t` is an estimate of `(gradient magnitude)²` per coordinate**. So `m̂_t / √v̂_t` is "the gradient direction, but with each coordinate rescaled so they're all on the same scale." Per-coordinate adaptive learning rate. Concretely: where a coordinate's gradient is consistently large, `m̂_t` and `√v̂_t` are both large and **cancel**, leaving a step of `≈ η`; where it's consistently tiny, small-over-small again gives `≈ η`. Adam takes a step of roughly size `η` in *every* coordinate, regardless of its raw gradient scale.

Read `m̂_t / √v̂_t` as a rough **signal-to-noise ratio**: consistent (low-variance) gradients give a ratio near `±1` and a confident step; noisy, sign-flipping gradients make `√v̂` exceed `|m̂|`, shrinking the step. Adam moves boldly where the gradient is reliable, cautiously where it's noisy. (The flip side: `β2 = 0.999` makes `v̂` react *slowly* to a sudden gradient spike — the denominator still reflects the old calm scale while the numerator explodes — which is one source of the loss spikes in file `05`, and why large-model training clips gradients and sometimes lowers `β2`.)

Why this matters: in DL, gradients have *wildly* different scales across parameters. The embedding matrix has different scales than the attention output projection has different scales than the bias vector. SGD with a single global LR has to pick a value that works for the largest-gradient parameter (else those updates blow up), which under-trains the small-gradient ones. Adam normalizes per-coordinate — every parameter gets an update of similar size regardless of its raw gradient magnitude.

**The limitation — Adam only scales the axes it's given.** `1/√v̂` is a *diagonal* preconditioner: one independent scale per coordinate. Geometrically, that can stretch or squash along the parameter axes (`θ_1, θ_2, ...`) but cannot **rotate**. The loss landscape's natural axes — the curvature (Hessian) eigen-directions — are generally *not* aligned with the parameter axes: picture a narrow valley running *diagonally* across your coordinates. Adam can rescale each axis but can't tilt to line up with the valley, so the off-axis ill-conditioning (the correlation *between* parameters) survives and the optimizer still zig-zags. Fixing it requires a *full-matrix* preconditioner that rotates into the landscape's natural axes, scales there, and rotates back — which is exactly what second-order methods like **Shampoo** (below) approximate cheaply. Adam keeps the diagonal approximation because it's cheap and captures *enough* of the conditioning to work well in practice.

**Per-parameter memory cost**: 2× the parameter count for `m` and `v`. So at fp32: 8 bytes per parameter (Adam state) on top of 4 bytes for the parameter itself. For a 70B model: 560 GB of Adam state. This is why optimizer state is a major training-memory consumer and why ZeRO-style sharding (Part 12) was developed.

## AdamW (Loshchilov & Hutter, 2017)

Adam plus a fix to how weight decay interacts with the optimizer.

In the original Adam (and in old PyTorch code), weight decay was implemented as an addition to the loss:
```
L_total = L_task + (λ/2) ‖θ‖²
```
The gradient of this term is `λ θ`, which gets folded into `g_t` and then normalized by the second moment. **This means weight decay's strength becomes per-coordinate-dependent**: parameters with large historical gradients have small effective decay (because `√v̂` is large, so the decay term is divided by a big number), and vice versa.

The fix in AdamW: **decouple** weight decay from the gradient update. Apply it as a separate, undivided shrinkage:
```
m_t, v_t, m̂_t, v̂_t as above
θ_{t+1} = θ_t - η · (m̂_t / (√v̂_t + ε) + λ · θ_t)
```

Or equivalently:
```
θ_{t+1} = (1 - η λ) · θ_t   -   η · m̂_t / (√v̂_t + ε)
          └── shrink ──┘         └──── learn ────┘
```

Read the update as **two independent operations on the same weights**: first *shrink* — multiply `θ` by `(1 - η λ)`, a number just below 1 (e.g. `1 - 3e-4·0.1 ≈ 0.99997`), nudging every weight slightly toward 0 — then *learn* — take the usual gradient step. `(1 - η λ)` is an explicit per-step shrinkage applied **directly to `θ`, not routed through the gradient**, so every parameter gets the same fractional shrinkage regardless of its second-moment magnitude. That separation is the whole point of the "W": in old Adam the decay was folded *into* the gradient, where the `1/√v̂` rescaling then distorted it per-coordinate; pulling it out into this clean multiplicative shrink is the fix.

Empirically: AdamW generalizes much better than Adam-with-L2-regularization on the same objective. The 2017 paper showed this; subsequent practice confirmed it across LLMs. **Every modern LLM trains with AdamW.**

## What weight decay actually does (briefly)

Weight decay shrinks every parameter toward 0 by a small fraction at every step. Two effects:
- **Regularization**: discourages large weights, which (heuristically) reduces model complexity and improves generalization.
- **Counteracts random drift**: without decay, weights can drift to large magnitudes during training as small random updates accumulate. Decay provides a restoring force.

Typical values: `λ = 0.1` for AdamW (note: this is the *AdamW* parameter, which already includes the `η` multiplier in the shrinkage; the *effective* shrinkage per step is `η · λ`).

**Don't apply weight decay to biases or norm parameters (`γ`, `β`).** These aren't capacity-carrying weights — decaying `γ` toward 0 drives a norm layer's output toward 0 (sabotage, not regularization), and decaying a bias just fights the offset the model learned. Neither benefits from the shrinkage. Modern code uses parameter groups:
```
decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
optimizer = AdamW([
    {'params': decay_params, 'weight_decay': 0.1},
    {'params': nodecay_params, 'weight_decay': 0.0},
])
```

The `p.dim() >= 2` filter is a heuristic: weight matrices are 2D+, biases and norm params are 1D. Crude but standard.

### Embeddings are a judgment call, not a clear "no"

Embeddings are often *also* named in "don't decay" lists, but it's genuinely a judgment call — and note the filter above puts the embedding matrix (2D, `V × D`) in the **decay** group. The tension comes from the lookup nature of embeddings:

- In a given batch, only rows for tokens that **appear** get a gradient; every other row gets zero. But decoupled weight decay shrinks **every row, every step**. So for an **untied** input embedding, a rare token's row is decayed on all ~100K steps but only gets a gradient to push back on the few steps its token shows up — the row **decays toward zero as a frequency artifact**, not because it's useless. That's the argument for *excluding* untied embeddings from decay.
- With **weight tying** (embedding = output/unembedding matrix, the common default), that matrix is also the output projection, so the full-vocabulary softmax produces a **dense** gradient into every row every step. The sparse/frequency-bias problem evaporates, decaying it behaves like decaying any weight matrix, and it usefully controls the logit scale. This is exactly why nanoGPT's `p.dim() >= 2` rule decays the (tied) embedding without issue.

Rule of thumb: **untied input embeddings** → lean *no decay*; **tied embeddings** → decaying is fine and common. Embedding magnitude isn't meaningless (it sets the token's push on the residual stream and the logit scale) — the real issue is whether its gradient is sparse, which tying fixes.

## Why Adam beats SGD for Transformers specifically

Three reasons:

1. **Gradient scale heterogeneity.** Different parameters in a Transformer (embeddings, attention projections, FFN matrices, norm parameters) have wildly different gradient magnitudes. Adam's per-coordinate normalization handles this; SGD can't.
2. **Sparse-ish gradients.** Embedding gradients are sparse (only embeddings for tokens in the batch get updated). Adam adapts the LR per coordinate, which helps these get useful updates. SGD applies the same global LR to everything.
3. **Loss landscape.** Transformers have noisy, non-isotropic loss landscapes. The second-moment normalization stabilizes the step direction.

The flip side: Adam's memory cost is 2× the params. For frontier-scale training, this becomes a major constraint.

## Variants you'll see

- **Adafactor** (Shazeer & Stern, 2018): factorizes the second-moment matrix to save memory. Used in T5 and some Google models. Roughly half the memory of Adam.
- **Lion** (Chen et al., 2023): uses only sign of momentum (`sign(m_t)`), no second moment. Half the memory of Adam, claims competitive quality. Some adoption.
- **Shampoo** (Anil et al., 2020): full-matrix preconditioner for second-order updates. Used in some Google production training. Expensive but converges faster per step.
- **Sophia** (Liu et al., 2023): diagonal Hessian-based optimizer. Claims faster convergence than Adam in pretraining.

None of these have displaced AdamW as the default for open-source LLM training. If you're not doing frontier scale, just use AdamW.

## Self-check

1. Why does Adam's per-coordinate second-moment normalization help when training Transformers, where parameters have wildly different gradient magnitudes?
2. What's the *exact* mathematical difference between Adam-with-L2-regularization (gradient of `L + (λ/2)‖θ‖²`) and AdamW? Why does the difference matter?
3. The bias correction `m̂_t = m_t / (1 - β1^t)` is large at small `t`. What pathology does this prevent in the first few steps of training?

### Answers

1. Embedding gradients can be ~100× smaller than FFN-matrix gradients (because embeddings are only updated for tokens that appeared in the batch). A global LR has to be small enough that the *largest*-gradient parameter doesn't explode, which means the smallest-gradient parameter takes 100× too-small steps. Adam normalizes each coordinate by `√v_t`, which is essentially the gradient magnitude. After normalization, every parameter sees a step of similar size (≈ `η` per dimension). Embedding updates are now comparable in size to FFN updates. SGD would either explode FFN or starve embeddings — can't get both.
2. **Adam with L2**: the L2 term contributes `λ θ` to `g_t`. This `λ θ` gets averaged into `m_t` and squared-averaged into `v_t`. The effective decay applied to parameter `i` is `λ θ_i / √v_i,t` — *divided* by the per-coord second moment. **AdamW**: the decay is `λ θ` directly subtracted from `θ` (after the Adam step), with no division by `v_t`. So in AdamW, every parameter has the same fractional shrinkage per step (`η λ`), regardless of its gradient history. The difference matters because Adam-with-L2 effectively *removes* the regularization for the parameters that have large historical gradients (they get tiny effective decay) and *amplifies* it for parameters with small historical gradients — exactly backwards of what you want. AdamW restores uniform regularization across parameters.
3. At `t = 1`, `m_1 = (1-β1) · g_1` (since `m_0 = 0`). With `β1 = 0.9`, `m_1 = 0.1 · g_1` — ten times smaller than the gradient itself. Without bias correction, the first step would be tiny, and the moving average would take ~`1/(1-β1) = 10` steps to "warm up" to the actual gradient scale. The bias correction `1/(1-β1^1) = 10` exactly cancels this initial damping. Same for `v_t`. Without bias correction, Adam takes ~100 steps to behave normally (because `β2 = 0.999` gives a much longer warmup time for the second moment); with correction, it's behaving sensibly from step 1.

## Exercise

In PyTorch, train two identical small Transformers on the same task, identical seed:
1. With `torch.optim.SGD(momentum=0.9, lr=1e-3)`.
2. With `torch.optim.AdamW(lr=1e-3, weight_decay=0.1)`.

Plot loss curves. AdamW should converge much faster, with smoother loss. Now try SGD with a 10× larger LR — it'll oscillate or NaN. Adam isn't magic, but for this problem class it's robustly better. This is why every LLM trains with AdamW.
