# Softmax, Log-Softmax, and Numerical Stability

## Softmax

For logits `x ∈ R^n`:
```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

In words: exponentiate each entry, then divide by the sum of all the exponentials. The `exp` makes every value positive; dividing by the total normalizes the outputs to sum to 1 — so the result is a valid probability distribution by construction.

Turns arbitrary real numbers into a probability distribution. Two common uses:
- **LM output layer**: convert vocab logits to token probabilities.
- **Attention**: convert QK scores to attention weights.

## Temperature

```
softmax_T(x)_i = exp(x_i / T) / Σ_j exp(x_j / T)
```

`T` is a knob that controls how *peaky* vs *flat* the distribution becomes — and it does this by rescaling the **gaps between logits** before they get exponentiated. Softmax only cares about *differences* between logits (any constant added to all of them cancels out in the ratio), so dividing every logit by `T` shrinks or stretches those differences:

- **`T > 1`** shrinks the gaps. A pair of logits that were 4 apart become 2 apart at `T=2`. Smaller gaps → softer ratios after `exp` → distribution flattens toward uniform. The model is "less confident."
- **`T < 1`** stretches the gaps. The same pair becomes 8 apart at `T=0.5`. Bigger gaps → `exp` blows the ratios way up → distribution peaks toward argmax. The model is "more confident."
- **`T → 0`**: gaps go to infinity, softmax collapses to a one-hot at the largest logit (pure argmax — deterministic).
- **`T = 1`**: standard softmax.
- **`T → ∞`**: gaps go to zero, softmax collapses to a uniform distribution.

The nonlinearity of `exp` makes this knob very sensitive: a 2× change in gap turns into a much-more-than-2× change in probability ratio. That's why small `T` changes in LM sampling produce large changes in output diversity.

In attention, the √d scaling is effectively a fixed temperature of `√d`. In sampling, temperature is a user-controllable knob.

## Why vanilla softmax is numerically unstable

`exp(x_i)` overflows to `inf` for `x_i > ~88` (float32). Underflows to 0 for `x_i < ~-88`. For `d_model=4096`, raw dot products can easily exceed 100+ without scaling — meaning vanilla softmax on unscaled attention scores returns NaN.

**Why dot products get that big.** A QK dot product is `q · k = Σ_{i=1}^{d_head} q_i k_i` — a sum over `d_head` terms (typically 64 or 128). If components are roughly unit-scale (which is what LayerNorm and standard init engineer for), each term is ~O(1), so the sum grows with `d_head`:

- **At init (random Q, K).** Components are ~`N(0, 1)`, so the dot product has mean 0 and **variance `d_head`** — standard deviation `√d_head`. For `d_head = 128`, typical scores are ~`±11`; the *max* across `S = 8192` positions scales like `√(2 log S) · √d_head ≈ ±50`. Already meaningful.
- **During training (Q, K aligned).** The model *learns* `Q` and `K` to align on relevant pairs. By Cauchy-Schwarz, `q · k ≤ ||q|| · ||k||`, and aligned vectors approach that bound. Component norms also drift above unit scale, especially in post-LN architectures. Aligned dot products easily reach 100–500.

**Why `√d_head` scaling fixes it.** Dividing by `√d_head` exactly cancels the variance growth: at init, `(q · k) / √d_head` has variance 1 regardless of head size. After training the aligned values still grow, but from a much lower base — `exp` can handle them without overflowing.

This is also why head dim is usually capped at 64–128: even with `√d` scaling, very large `d_head` makes the post-softmax distribution too sharp (relevant SNR scales with `√d_head`), and you lose the soft-attention dynamic. Modern Transformers scale up by adding *more heads*, not bigger ones.

**Aside — softmax sharpness is not the reason for multi-head.** It constrains `d_head` (per-head dim), not `H` (number of heads). The actual reason you want `H > 1`: **a softmax produces one peak, so one head can attend to one thing per query.** Real tokens care about many things in parallel (previous token, clause subject, matching bracket, coreferent mention, BOS sink). You cannot express "A and B and C" inside one softmax — it's a soft argmax, it picks. So you run `H` softmaxes in parallel over `H` different `(W_Q^h, W_K^h, W_V^h)` subspaces. Secondary benefits: the multi-head construction is rank-`d_head` per head summed through `W_O` — strictly more expressive than one rank-`D` head at equal parameter count; and heads specialize into distinct subspaces of the residual stream (induction heads, previous-token heads, name-mover heads — see Part 11.2). Evidence the sharpness story isn't load-bearing: MQA/GQA collapse the K/V side but keep many *query* heads — pattern multiplicity is what matters. If softmax were replaced by a linear kernel, you'd still want multi-head; if QK-norm let `d_head` grow arbitrarily, you'd still want multi-head. Full treatment in Part 5.1.

## The log-sum-exp trick

Subtract `max(x)` before exponentiating:
```
softmax(x)_i = exp(x_i - m) / Σ_j exp(x_j - m)    where m = max(x)
```

Mathematically identical (top and bottom both scale by `exp(-m)`). Numerically: the largest exponent is now `0`, so `exp(0) = 1` is the biggest thing you compute. No overflow. Underflow of small exponents still happens but is harmless (they contribute ~0 to the sum).

This is why you'll see `x - x.max()` littering stable softmax implementations, including in FlashAttention's online algorithm (which maintains a running max across tiles).

## Log-softmax

```
log_softmax(x)_i = x_i - logsumexp(x)    where logsumexp(x) = log(Σ exp(x_j))
```

This is just the log of softmax with the algebra carried out: `log(exp(x_i) / Σ_j exp(x_j)) = log(exp(x_i)) - log(Σ_j exp(x_j)) = x_i - logsumexp(x)`. The numerator collapses because `log` and `exp` are inverses; the denominator can't, so it stays as `logsumexp`. Same normalizing structure as softmax, transported to log-space — division by `Σ exp` becomes subtraction of `log(Σ exp)`.

`logsumexp` is itself a *soft max* — a smooth approximation of `max(x)`. So read `x_i - logsumexp(x)` as "how much does logit `i` exceed the (soft) maximum?" — which is `0` for the winner and increasingly negative for losers. That's exactly what log-probabilities should look like: `0` for certain, `-∞` for impossible.

Compute `logsumexp` stably as `m + log(Σ exp(x_j - m))`.

**Prefer `log_softmax` + `nll_loss` over `log(softmax(x))`** — the direct form avoids a round-trip through the small numbers that underflow in the log.

In PyTorch: `F.cross_entropy(logits, targets)` internally does `log_softmax` + `nll_loss` fused. Use it. Never manually compose `softmax` → `log` → `gather`.

## The softmax Jacobian (worth memorizing)

Let `s = softmax(x)`. Then:
```
∂s_i/∂x_j = s_i (δ_{ij} - s_j)
```

Equivalently: `J = diag(s) - s s^T`.

Key consequence: when combined with cross-entropy loss, the gradient simplifies beautifully:
```
∂L/∂x = s - y     where y is one-hot target
```

That's it — the gradient of CE-on-softmax w.r.t. logits is just `(predicted_probs - target_probs)`. This elegance is *why* CE+softmax is the standard combo; it's not a coincidence.

**The intuition, without the algebra.** Softmax on its own has a problem: when it confidently predicts something (output near 0 or near 1), its sensitivity to input changes flattens out — that's saturation. If the model is confidently *wrong*, gradients shrink to nothing and it gets stuck. Cross-entropy has the opposite shape: the more confidently wrong you are, the more sharply it punishes you, and the more its derivative grows. When you stack them, the two shapes exactly cancel — softmax's flattening at the extremes is undone by cross-entropy's blowing-up at the extremes. The combined gradient stays strong everywhere, including in the regime where the model is confidently wrong and most needs a wake-up call.

What's left after the cancellation is the simplest possible gradient: **predicted probabilities minus target probabilities**. Just the error. No saturation curve baked in, no Jacobian factors, no scaling tricks. The signal that flows backward into the network is literally "here's how wrong you were on each class."

And this isn't a happy coincidence. There's a deeper principle from the theory of generalized linear models: every output activation has a *matching* loss whose derivative cancels the activation's sensitivity curve, and the resulting gradient is always observed-minus-predicted. Sigmoid pairs with binary cross-entropy the same way. Softmax pairs with multi-class cross-entropy. The loss was built to match the activation, and you can see it in how cleanly everything collapses.

This is also why you'll never see MSE (mean squared error — the average of squared differences between prediction and target, the standard regression loss) paired with softmax in practice. MSE doesn't have the right shape to cancel softmax's saturation curve, so confidently-wrong predictions would produce vanishing gradients and the model would get stuck. Cross-entropy was the right partner all along.

## Saturation

When one logit dominates, `softmax` produces ~1 for that entry and ~0 elsewhere. The Jacobian `s(1 - s)` then vanishes → gradient is ~0 → the model can't learn out of a wrong confident prediction. This is:
- Why you scale attention scores by `√d` (to prevent pre-softmax saturation).
- Why label smoothing helps (keeps targets away from hard 0/1).
- Why temperature > 1 during exploration in RL.

## Self-check

1. If attention scores pre-softmax are `[10, 10.5, 100]`, what does naive softmax return? What does stable softmax return?
2. Derive `∂(softmax(x))_i / ∂x_j` from the definition.
3. Why is it better to take the log of softmax in one step (`log_softmax`) rather than compute softmax first and then log it?

### Answers

1. **Naive**: `e^100 ≈ 2.7 × 10⁴³`, which **overflows fp32** (max ≈ 3.4 × 10³⁸). Result: `inf` in the numerator, `inf/inf = nan` after normalization. NaN propagates and trashes the rest of the computation. **Stable** (subtract `max = 100`): exponents become `[-90, -89.5, 0]`, so `e^{-90} ≈ 0`, `e^{-89.5} ≈ 0`, `e^0 = 1`. Sum ≈ 1. Result: `[~0, ~0, ~1]`. Numerically clean. The math is identical (subtracting a constant from all logits doesn't change softmax) — only the float representation matters.
2. Quotient rule (full derivation): `s_i = e^{x_i} / S` where `S = Σ e^{x_j}`. `∂e^{x_i}/∂x_j = δ_{ij} e^{x_i}`. `∂S/∂x_j = e^{x_j}`. So `∂s_i/∂x_j = (δ_{ij} e^{x_i} S - e^{x_i} e^{x_j}) / S² = (e^{x_i}/S)(δ_{ij} - e^{x_j}/S) = s_i (δ_{ij} - s_j)`. Matrix form: `J = diag(s) - s s^T`.
3. Two reasons. **Numerical**: `log_softmax(x) = x - logsumexp(x)` works in log space throughout, avoiding the underflow that happens when small probabilities round to 0 and `log(0) = -inf`. The composed `log(softmax(x))` round-trips through small probabilities and loses precision. **Efficiency**: skips the unnecessary `exp` → `log` round trip. PyTorch's `F.cross_entropy` fuses `log_softmax + nll_loss` for both reasons. Never write `log(softmax(x))` manually.

## Exercise

Implement `stable_softmax(x)` by hand using the max-subtraction trick. Verify against `torch.softmax` on inputs that would overflow (e.g. `[1e3, 1e3+1, 1e3-1]`). Then implement `stable_log_softmax` and verify on inputs with large negative values.
