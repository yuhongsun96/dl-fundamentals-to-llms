# Activations: ReLU → GELU → SwiGLU

The pointwise nonlinearity sandwiched between linear layers. Looks like a trivial choice — it isn't. Activation shape controls gradient flow, dead-neuron rates, and ultimately whether deep stacks train at all. The activation history of NLP is short and worth memorizing.

## What an activation actually has to do

Two jobs, looked at from opposite ends:

- **Forward**: bend the linear function so stacking layers expresses something nonlinear (file `01`). Without a bend, depth is useless.
- **Backward**: pass a useful gradient. The backward signal through `σ` is `σ'(z)` multiplied elementwise with the upstream gradient. If `σ'(z) = 0` in some region, gradients **die** through that region — the unit stops learning.

Every design decision below trades off these two demands.

## The design pressure: add the *least* nonlinearity that works

It's tempting to think the job is to *add* nonlinearity, so a richly nonlinear function would be best. It's the opposite. Model capacity lives in the **linear maps** — that's where the parameters are; the activation only has to stop consecutive linear layers from collapsing into one. So the activations that win are the ones that **distort the signal as little as possible** — "as linear as you can get away with." A near-linear network has a smoother loss landscape and optimizes more easily; strongly nonlinear per-layer functions (`sin`, `x³`, sharp bumps) make the composed function wildly curved and training hard. This reframes the whole file: the goal is *minimal* nonlinearity, not maximal.

Two hard constraints then eliminate most candidate shapes *before* any experiment:

- **Gradient flow.** Backprop multiplies `φ'(z)` layer after layer, so `φ'` must be **≈ 1 over a broad, commonly-visited region** and must never blow up. This kills bounded-both-sides functions (tanh/sigmoid → vanish), unbounded polynomials (`x³` → explode), and localized bumps (RBF → ~zero gradient almost everywhere). What survives is the **near-linear, monotone, one-side-controlled** shape: a slope-1 region for the signal, a bounded/suppressed region for stability. The result is a roughly **piecewise-linear network** — a universal approximator with the most benign geometry available.
- **Scale-robustness.** ReLU is positively homogeneous (`ReLU(αx) = α·ReLU(x)` for `α > 0`) — no preferred input scale, which is exactly why it composes cleanly with normalization and variance-preserving init. Functions with a built-in scale (tanh's knee, a bump at a fixed location) force you to keep activations pinned at that magnitude.

Within the surviving family, the *specific* pick is **empirical and low-stakes** — and notably, the *amount* of negative suppression barely matters: ReLU (negatives → 0), LeakyReLU (→ `0.01x`, almost no gating), ELU, and GELU/SiLU all train about equally well. When Ramachandran et al. (2017) ran an architecture *search* over activation functions with no ReLU prior, it rediscovered `x·σ(βx)` (Swish/SiLU) — strong evidence this region of function-space is genuinely good, not just entrenched. So read the specific shapes below as points in one family that all satisfy the two constraints; the differences between them are refinements (chiefly *smoothness*, for gradient flow near zero), not deep principles.

## tanh and sigmoid — the old guard

```
sigmoid(x) = 1 / (1 + e^{-x})            range (0, 1)
tanh(x)    = (e^x - e^{-x}) / (e^x + e^{-x})    range (-1, 1)
```

These dominated the 1990s and the early-2010s LSTM era. Both are **smooth and bounded**. Both **saturate at the tails** — for large `|x|`, the slope is ~0.

The problem is gradient vanishing in deep networks. Stacking 10 sigmoid layers means multiplying 10 slopes each `≤ 0.25` (the max of sigmoid's derivative); upstream gradients shrink by `0.25^10 ≈ 10^{-6}` before reaching the first layer. Training stalls.

You still see sigmoid/tanh inside LSTM cells (for gates), and sigmoid as the *output* nonlinearity in binary classifiers and gating mechanisms. You won't see them as the FFN activation in any modern stack.

## ReLU — the simplification that fixed everything

```
ReLU(x) = max(0, x)
```

Two pieces: 0 for negative inputs, identity for positive. Trivial to compute. Trivial to differentiate: `ReLU'(x) = 1[x > 0]`.

Why it broke the dam in 2010s deep learning:
- **No saturation on the positive side.** Gradient is exactly 1 for any `x > 0` — no shrinking factor multiplying through the chain rule.
- **Sparse activations.** Roughly half of pre-activations are negative at init, giving exact zeros downstream. Sparsity is cheap to compute and (sometimes) acts as implicit regularization.
- **Cheap.** A `max` is one instruction; the derivative is a comparison.

This is what made training >10-layer networks routine.

### Dead ReLUs

The flip side: `ReLU'(x) = 0` for `x < 0`. If a unit's pre-activation goes negative *for every input in the dataset*, the gradient through it is exactly 0, the weights stop updating, and the unit is **dead** — permanently outputting 0, never recovering. Common cause: a too-large learning rate pushing a unit into a bad region. Once dead, a fraction of your capacity is just gone.

Mitigations (in chronological order of popularity): LeakyReLU (`max(0.01 x, x)`), PReLU (learnable leak), ELU. They all give a nonzero gradient in the negative region so dead units can recover. None became standard — the next class of activations made the problem mostly irrelevant.

## GELU — what BERT and GPT-2 used

```
GELU(x) = x · Φ(x)
```

where `Φ` is the standard Gaussian **CDF** (cumulative distribution function). `Φ(x)` answers one question: *what fraction of a standard bell curve (mean 0, unit spread) lies to the left of `x`?* It accumulates area from `−∞` up to `x`, so it rises smoothly from `0` (far left), through `0.5` (at `x = 0`, half the curve sits below center), to `1` (far right) — an S-shaped 0→1 curve. Read it as a **soft gate in `(0, 1)`** — "how much of `x` should pass through?" — which makes `GELU(x) = x · Φ(x)` read as "keep a fraction of `x` proportional to how confidently positive it is."

The true `Φ` has no elementary closed form (it needs the error function `erf`, which is slow), so implementations use a cheap `tanh`-based approximation tuned to trace the same S-curve:
```
GELU(x) ≈ 0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))
```

Two ways to read it:
- **Stochastic gating.** "Multiply `x` by the probability it's positive under a unit Gaussian." For large positive `x`, `Φ(x) → 1`, output `≈ x` (ReLU-like). For large negative `x`, `Φ(x) → 0`, output `≈ 0`. The transition is **smooth**.
- **Soft ReLU.** A ReLU with a curved transition through 0 instead of a kink, with a small negative dip for small negative inputs.

Why it beat ReLU empirically:
- **Smooth derivative everywhere.** No kink at 0, no exact zero for `x < 0`. Gradients flow even from slightly-negative pre-activations. Dead-neuron problem softened.
- **Non-monotone in the negative region** (small negative dip near `x = -1`), which seems to help — though no one has a fully satisfying theoretical story.

**"But `Φ` is basically a sigmoid — doesn't GELU vanish gradients like sigmoid did?"** No. `Φ` and `σ` are near-identical S-curves, but GELU returns `x · Φ(x)`, not `Φ(x)` — the same `x·(gate)` move that turns `σ` into SiLU below. Multiplying by `x` makes GELU unbounded on the positive side, so `GELU'(x) → 1` as `x → +∞` (the output there is `≈ x`, an identity). Sigmoid-*as-activation* instead has derivative `≤ 0.25` everywhere and `→ 0` on both tails, so a deep stack multiplies many `≤ 0.25` factors and vanishes. A stack of active GELU units multiplies factors near 1 — no systematic decay. GELU only zeroes its gradient on the *deep-negative* side (ReLU's "off" region), never on both tails.

**And that deep-negative zero doesn't cause dead units the way ReLU's does.** A ReLU dies because its gradient is *exactly* 0 for all `x < 0` — a flat plateau, a one-way door: a unit stuck negative gets zero update and can never climb back. GELU has no flat region. Its negative-side derivative `Φ(x) + x·φ(x)` (with `φ` the Gaussian bump) is genuinely nonzero through the moderate range — it even dips *negative* near `x ≈ −1` — and only fades to ~0 in the deep tail (`x ≲ −4`). So a unit at a mildly-negative pre-activation still receives a restoring nudge and can recover; being "as good as dead" would require the deep-negative tail for *every* input at once, far stricter than ReLU's "merely negative for every input." Smoothing the kink is exactly what removes the trap.

GELU was the default for BERT, GPT-2, GPT-3, T5, and most encoder/decoder models through 2021-ish. Still the default if you're using HuggingFace's stock `BertModel`.

## SiLU / Swish — the bridge

```
silu(x) = x · σ(x)    where σ is sigmoid
```

Also called **Swish** in the original Google paper. Look familiar? It's GELU's cousin: GELU multiplies by a Gaussian CDF, SiLU multiplies by a sigmoid (a logistic CDF). The shapes are nearly identical — same dip, same smooth transition, same right-tail-linear behavior. SiLU is slightly cheaper to compute (no `tanh` or `erf`).

By itself, SiLU isn't a meaningful upgrade over GELU. What it enabled is the gated form below.

**Aside — why GELU exists, and why not just use SiLU everywhere.** Both are the same family: **Swish**, `x · σ(β x)`. SiLU is `β = 1`; GELU matches `β ≈ 1.702` (since `Φ(x) ≈ σ(1.702 x)`), so GELU is a slightly *sharper*-gated Swish — close to SiLU, not identical. GELU won the early Transformer era largely by timing: it came first (2016), arrived with a principled probabilistic story (a smooth stochastic 0/1 gate, dropout's cousin), and got baked into BERT and GPT-2 — so everything downstream copied it (HuggingFace's `BertModel` still defaults to it). SiLU's lower compute is real but negligible: the activation is a pointwise op dwarfed by the FFN's matmuls, so it never justified switching a working model. The tell that the choice is within noise: the frontier makes it *both* ways today — **SwiGLU** (SiLU-gated: Llama, Mistral, Qwen, DeepSeek) and **GeGLU** (GELU-gated: PaLM, Gemma) are both top-tier. What mattered was *gating*, not whether the gate is a Gaussian or a logistic CDF.

### "Isn't this just the sigmoid from above, times x?"

Algebraically, **yes** — `silu(x)` is literally the same `σ(x)` from the old-guard section, multiplied by the input `x` instead of returned directly. But that single multiply turns it into a function that behaves *nothing* like sigmoid as an activation. Read `σ(x)` not as a squashing output but as a **soft gate in `(0, 1)`** — "what fraction of `x` should pass through?" — and SiLU as `x` scaled by its own gate (a **self-gating** function). Three consequences:

| Property | `σ(x)` returned directly | `silu(x) = x·σ(x)` |
|---|---|---|
| **Range** | bounded `(0, 1)` | unbounded above |
| **Large positive `x`** | saturates → 1, slope → 0 (kills gradients) | `σ→1`, so output `→ x` — linear, identity-like, **no saturation** |
| **Large negative `x`** | saturates → 0 | `σ→0` fast, so output `→ 0` |
| **Monotonic?** | yes | **no** — dips slightly negative, min ≈ `−0.28` near `x ≈ −1.28` |

The middle row is the whole point. Sigmoid-as-activation saturates on *both* tails, so deep stacks vanish (that's the 1990s gradient problem in the old-guard section). Multiplying by `x` removes the positive-side saturation entirely — for confident-positive inputs SiLU just passes `x` through like ReLU — while the gate still smoothly suppresses negatives. So it keeps sigmoid's smooth, differentiable transition but discards the bounded-squashing behavior that made plain sigmoid unusable as an FFN nonlinearity.

So: same formula component, completely different job. `σ(x)` alone is an output-squasher / probability / LSTM gate; `x·σ(x)` is a non-saturating self-gated activation in the ReLU/GELU family.

**How big do these outputs actually get in a forward pass?** The pre-activation `x` (one hidden coordinate of `x̂ W_up`) is typically O(1): with unit-RMS input and variance-preserving init each coordinate is roughly `~N(0, 1)`, so `|x| > 1` about a third of the time — and more as weights grow during training, since nothing normalizes `x` directly (the norm sits *before* the projection). The output `x·σ(x)` is **unbounded above** and clears 1 once `x ≳ 1.28` (`silu(2) ≈ 1.76`, `silu(3) ≈ 2.86`, and `→ x` for large `x`), so the strongly-positive nodes routinely exceed 1 — that is the non-saturation working as intended, not a bug. The negative side is bounded (min ≈ `−0.28`), so outputs never swing large-negative. Beware that real trained LLMs also develop rare **outlier / "massive activation"** channels reaching into the hundreds; that growth on the positive tail is part of why stabilizers like QK-norm and logit soft-capping (Part 3.2) exist.

## SwiGLU — what every modern LLM actually uses

Take an FFN and replace the single linear → nonlinearity → linear with a **gated** structure:

```
SwiGLU FFN(x) = (silu(x W_gate) ⊙ (x W_up)) W_down
```

In words: project `x` two ways. Pass one through SiLU. Multiply the two projections elementwise (`⊙` is Hadamard). Project the product back down.

The gating is the trick. SiLU on its own is a pointwise function — its expressive power per element is limited. Multiplying it elementwise with a parallel learned projection lets each output coordinate *modulate itself*: the gate `silu(x W_gate)` decides how much of `x W_up` to let through, per-feature, per-position. This is the same trick as GLU (Gated Linear Unit) and its variants (GeGLU = GELU-gated, ReGLU = ReLU-gated). SwiGLU just happens to be the variant that won.

A 2026 refinement, **SiTU-GLU** (Sigmoid Tanh Unit GLU, Kimi K3), keeps SwiGLU's gate but wraps both the gate's linear factor and the up branch in a smooth soft-cap `softcap(x, β) = β·tanh(x/β)`:
```
SiTU-GLU(x) = ( softcap(x W_gate, β₁) ⊙ σ(x W_gate) ) ⊙ softcap(x W_up, β₂)      β₁ = 4, β₂ = 25
```
Near the origin `β·tanh(x/β) ≈ x`, so it matches SwiGLU locally; for large inputs each factor saturates to `±β`, bounding the output by `β₁·β₂ = 100` where plain SwiGLU is unbounded. It's a direct fix for the unbounded positive tail / massive-activation growth flagged above — introduced to stabilize an extreme-sparsity MoE (896 experts, 16 active per token), alongside an RMSNorm before the up-projection.

### Why three weight matrices?

**The classic FFN has two.** Project up, bend, project down:
```
x → (W_in) → widen to d_ff → nonlinearity → (W_out) → shrink back to D
```
Traditionally `d_ff = 4D` (inner layer 4× wider than the model dim). Two matrices, each `D × d_ff`, so `2 · D · d_ff` params: `W_in ∈ R^(D, d_ff)`, `W_out ∈ R^(d_ff, D)`.

**SwiGLU adds a third for the gate.** Instead of one up-projection it makes *two in parallel*, then multiplies them:
```
x → W_gate → SiLU ┐
                  ⊙ → W_down → back to D
x → W_up  ────────┘
```
- `W_gate` → the **gate**: fed through SiLU, a per-feature "how much should pass through?" signal.
- `W_up` → the **value**: the content to be passed.
- Multiply elementwise (gate modulates value), then `W_down` projects back to `D`.

So three matrices: `W_gate, W_up ∈ R^(D, d_ff)`, `W_down ∈ R^(d_ff, D)`. The extra matrix (`W_gate`) is exactly what buys the gating mechanism. But naively, three matrices at the old `4D` width is `3 · D · d_ff` params — **1.5× bigger** than the classic block. Comparing that to a 2-matrix FFN isn't fair: you'd just be adding parameters.

**The 2/3 trick — shrink the width to pay for the third matrix.** To keep the block the *same size* (so you measure the architecture, not the extra params), SwiGLU narrows the inner dimension. Simple bookkeeping:
```
3 · d_ff_new = 2 · (4D)        ← three skinny matrices cost what two fat ones did
d_ff_new = (2/3) · 4D ≈ 2.67 D
```
Set the inner width to **two-thirds** of the classical `4D` and three skinnier matrices cost the same as two fatter ones. The gating comes essentially free in params and FLOPs — the win is expressivity *per parameter*, not more parameters. (Self-check #3 below works the same compensation through the FLOP count: `8D²` either way.)

**The Llama-2 7B numbers, decoded.** `D = 4096`. Classical `4D` width would be `16384`; two-thirds of that is `≈ 10923`. They round to `d_ff = 11008` (very close to `8/3 · D = 10922.7`), nudged to a hardware-friendly multiple. So that oddly specific `11008` is just "`4×` width, scaled by `2/3` to absorb the gate matrix, rounded for the GPU."

### Why it actually helps

Empirical: SwiGLU beats ReLU/GELU FFNs at matched parameter count and matched compute. The Noam Shazeer paper ("GLU Variants Improve Transformer", 2020) is mostly an empirical sweep showing this; it has the honest tagline "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."

The folk explanation: multiplicative interactions are expressive in a way summation isn't. The gate-times-value structure lets the FFN learn input-dependent feature selection without needing depth — a single SwiGLU layer can route information conditionally, where a non-gated layer would need to learn the conditional behavior implicitly across two layers.

## Where they live now

| Activation | Used in |
|---|---|
| `tanh`, `sigmoid` | LSTM gates, binary heads. Not in FFNs. |
| `ReLU` | Older code, some efficient inference work. Rare in new LLMs. |
| `GELU` | BERT, GPT-2, GPT-3, T5, OPT. The "old default" for Transformers. |
| `SiLU / Swish` alone | Some non-Transformer models (EfficientNet, etc.). |
| `SwiGLU` | Llama 1/2/3, Mistral, DeepSeek, Qwen, Gemma — basically every modern open LLM. |
| `GeGLU` | PaLM, some Google models. Same idea as SwiGLU with GELU as the gate nonlinearity. |
| `SiTU-GLU` | Kimi K3 (2026). Soft-capped SwiGLU — `β·tanh(·/β)` on gate and up branches to bound activation growth at extreme MoE sparsity. |

## Self-check

1. Why does `tanh` struggle in a 50-layer network where ReLU works? Quantify the gradient shrinkage roughly.
2. What's the difference between "dead ReLU" and "saturated sigmoid"? Both involve a region of zero gradient — why is one fatal and the other manageable?
3. SwiGLU triples the FFN's matrix count (gate + up + down) but only doubles compute relative to a standard FFN. Why isn't compute also tripled?

### Answers

1. `tanh'(x) = 1 - tanh²(x)`, peaks at 1 at the origin, falls off fast. Typical activations after init sit around `tanh(z)` for `z ~ N(0, 1)`, so `tanh'(z)` averages around 0.4–0.5. Stacked through 50 layers: gradient is multiplied by ~`0.45^50 ≈ 10^{-17}`. Effectively zero — first-layer weights get no signal. ReLU has slope exactly 1 on the active region, so the product is 1 anywhere all units are active; even with half-dead activations the shrinkage is `0.5^50 ≈ 10^{-15}` *only for the dead paths*, while alive paths see no shrinkage at all. Crucially, residual connections (Part 3) bypass this entirely — but tanh's shrinkage problem is what made deep nets infeasible *before* skip connections.
2. **Dead ReLU**: a specific unit's pre-activation is below 0 for every input. The unit's weights never receive gradient, so they never change, so the pre-activation never moves back above 0. Permanent — the unit is gone for the rest of training. **Saturated sigmoid**: a unit's pre-activation is large (positive or negative), so its derivative is tiny but **nonzero**. Gradient still trickles through, slowly, and the unit can recover when the input distribution shifts. Sigmoid saturation slows learning; dead ReLU kills capacity. GELU/SiLU avoid both: smooth everywhere, no exact zero, no asymptotic saturation on the positive side.
3. The third matrix (`W_gate`) is `D × d_ff`, same shape as `W_up`. With `d_ff` set to `(2/3) · 4D` instead of `4D`, the per-matrix cost shrinks by 2/3. Total: `3 matrices · (D · (2/3) · 4D) = 8 D²` FLOPs (one direction). Standard FFN with `d_ff = 4D` and two matrices: `2 · 4 D² = 8 D²`. Exactly matched. The "3 matrices but only 2× compute" framing is just the 2/3 width compensation working as designed. Parameters and FLOPs both end up comparable to a 4D-width vanilla FFN. The win is purely expressivity-per-FLOP.

## Exercise

Plot ReLU, GELU, SiLU, and their derivatives on `[-5, 5]`. Note where each derivative is zero, where it's bounded above by 1, and where it exceeds 1 (it can!). For SwiGLU specifically, plot the 1D map `f(x) = silu(x) · x` (the per-element behavior with `W_gate = W_up = I`) and notice the asymmetric shape — that's the gating in action.
