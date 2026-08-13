# Approximations and Orders of Magnitude

Most quantitative claims in deep learning are approximations stated without their error bars: *"to first order the variance is preserved," "cosine similarity is `O(1/√D)`," "loss scales as a power law."* These are not sloppiness — they're a compressed notation that assumes you know which term was dropped and when dropping it stops being safe. This file is that reading skill.

## Taylor expansion: the only approximation tool you need

Any smooth function near a point `a` can be written as a polynomial in the displacement:

```
f(a + x) = f(a) + f'(a)·x + ½f''(a)·x² + (1/6)f'''(a)·x³ + …
```

Truncating after the linear term is **first order**; keeping the `x²` term is **second order**. The dropped terms shrink like the next power of `x`, so **the approximation is good exactly when `x` is small** — and "small" always means small relative to something.

The two cases that come up constantly, expanded around 0:

```
sin(x) = x − x³/6 + …        →   sin(x) ≈ x         (error ~x³)
cos(x) = 1 − x²/2 + …        →   cos(x) ≈ 1 − x²/2   (error ~x⁴)
```

### When "small" stops being small

| `x` | `sin x` | rel. error of `sin x ≈ x` | rel. error of `cos x ≈ 1 − x²/2` |
|---|---|---|---|
| 0.001 | 0.001000 | 0.00002% | 4×10⁻¹²% |
| 0.01 | 0.010000 | 0.0017% | 4×10⁻⁸% |
| 0.1 | 0.099833 | 0.17% | 0.0004% |
| 0.5 | 0.479426 | 4.3% | 0.29% |
| 1.0 | 0.841471 | **18.8%** | **7.5%** |

Read the pattern: the first-order error grows like `x²` (each 10× in `x` costs 100× in error), so the approximation is superb below 0.1, usable to ~0.5, and misleading by `x = 1`. Note also that the *second-order* `cos` approximation is far more accurate at every `x` — extra orders buy a lot.

Concretely, this is why the slowest sinusoidal positional dimensions are described as "a linear ramp plus a constant": their angle stays under `0.068` radians across 512 positions, where `sin x ≈ x` is accurate to better than 0.01% ([5.3 supplementary](../../part5_transformer_rebuilt/5.3_positional_information/supplementary/02_why_sinusoids.md)).

### First-order thinking elsewhere

- **Linearization.** A Jacobian *is* the first-order Taylor coefficient of a vector function — which is why backprop, built entirely from Jacobians, is a first-order method ([1.1/08](../1.1_linear_algebra/08_jacobians_chain_rule.md)).
- **Gradient descent** takes a first-order step; **second-order** methods add the Hessian, the quadratic term ([file 03](03_bilinear_and_quadratic_forms.md)).
- **"To first order, X doesn't matter"** means: the effect exists but enters at second order or beyond, so it's negligible for small perturbations. It is *not* a claim that the effect is zero — and a common source of surprise when perturbations stop being small.

## Reading `O(·)` and "scales like"

In ML writing, `O(f(D))` is used loosely to mean **"grows proportionally to `f(D)`, ignoring constants"** — not the formal worst-case complexity sense. The point is always the *shape* of the dependence.

The ones you need on sight:

| Expression | Meaning | Where it appears |
|---|---|---|
| `O(1)` | doesn't grow with the dimension | "gradient norms are `O(1)`" — bounded regardless of model size |
| `O(1/√D)` | shrinks slowly; halves when `D` quadruples | random-vector cosine similarity; typical entry of a unit vector |
| `O(√D)` | grows slowly | norm of a vector with `O(1)` entries |
| `O(D)` | linear | FFN cost per token in `d_ff` |
| `O(S²)` | quadratic | attention cost in sequence length — the reason long context is hard |

The reason `1/√D` shows up everywhere is worth internalizing: sums of `D` independent random signs grow like `√D` rather than `D`, because they partially cancel. Divide by `D` to average and you get `1/√D`. That single fact underlies near-orthogonality in high dimensions ([1.1/06](../1.1_linear_algebra/06_projections_orthogonality.md)), the `1/√d_k` attention scaling ([5.1/02](../../part5_transformer_rebuilt/5.1_self_attention/02_scaled_dot_product_attention.md)), and the `1/d_in` initialization rule ([2.3/01](../../part2_neural_network_fundamentals/2.3_init_normalization/01_init_variance_preservation.md)).

**Reading tip:** when you see a scaling claim, ask *"what is being held fixed?"* `O(S²)` attention cost holds the model fixed and grows the sequence; the same computation is `O(D)` if you hold the sequence fixed and grow the width. The exponent is a statement about one variable at a time.

## Orders of magnitude and log scales

An **order of magnitude** is a factor of 10. Working in orders of magnitude — rather than exact numbers — is the normal register for capability claims, and log scales are how you plot them.

Two rules for reading log plots:

- **A straight line on a log-*y* axis is exponential** growth or decay. Equal vertical steps = equal *multiplicative* factors. Used for loss curves, singular-value spectra ([file 02](02_energy_and_spectra.md)).
- **A straight line on log-*x* and log-*y* is a power law**, `y = c·x^α`, and the **slope is the exponent `α`**. This is the entire visual grammar of scaling laws: "loss falls as a power law in compute" means the log-log plot is a line, and the slope is the thing being reported.

Why power laws matter operationally: a power law with a small exponent means **diminishing but never-vanishing** returns. You always improve by scaling, and you always improve less per dollar. That shape is why scaling-law extrapolation is possible at all — and why nobody expects a single scaling axis to be sufficient.

The other reason log scales dominate: geometric spacing is the natural way to cover a wide range with few samples. It's the same reasoning behind geometrically-spaced positional frequencies ([file 08](08_frequency_phase_and_periodicity.md)) and behind sweeping learning rates as `1e-3, 3e-4, 1e-4` rather than in equal steps.

## Why this matters

- **Variance-preservation** arguments in initialization are first-order arguments; knowing that tells you when they break (deep stacks, large activations).
- **Scaling-law papers** are unreadable without log-log fluency, and Part 8's compute-optimal discussion assumes it.
- **`1/√D` reasoning** is the backbone of attention scaling, init, and high-dimensional geometry.
- **Spotting overreach.** "To first order this is fine" is a bounded claim. Asking "how big is `x` here, and what's the next term?" catches a large fraction of plausible-sounding but wrong arguments.

## Self-check

1. `sin(x) ≈ x` has relative error 0.17% at `x = 0.1` and 18.8% at `x = 1.0`. Why did a 10× increase in `x` cost ~100× in error?
2. What does "to first order, weight decay doesn't change the gradient direction" claim — and what does it *not* claim?
3. Two random unit vectors in `R^4096` have cosine similarity `O(1/√D)`. Roughly what number is that, and what happens if `D` grows to 16384?
4. A loss curve is a straight line on a log-log plot with slope `−0.05`. What functional form is that, and what does the small exponent imply operationally?
5. Someone reports attention is `O(S²)` and concludes it dominates cost. What should you check?

### Answers

1. Because the leading dropped term is `x³/6`, so the *absolute* error scales like `x³` while the value itself scales like `x` — the **relative** error therefore scales like `x²`. Ten times larger `x` gives ~100× the relative error: `0.0017% → 0.17% → 18.8%` as `x` goes `0.01 → 0.1 → 1.0`, which is what the table shows. Any first-order approximation has this quadratic error growth, which is why "small" needs to be checked numerically, not assumed.
2. It claims the **first-order effect** of decoupled weight decay is a pure radial shrink `W ← (1 − ηλ)W`, applied independently of the gradient term, so the descent direction from the loss is unaffected at that order. It does **not** claim decay has no effect on the trajectory — over many steps the shrink changes *where* you are, hence which gradients you subsequently see, and the interaction with adaptive optimizers is exactly why AdamW's decoupling was needed in the first place ([2.4/01](../../part2_neural_network_fundamentals/2.4_optimization/01_sgd_to_adamw.md)). "First order, per step" and "in aggregate, over training" are different claims.
3. `1/√4096 = 1/64 ≈ 0.016`. At `D = 16384` it becomes `1/128 ≈ 0.008` — quadrupling `D` halves the typical cosine. Slow decay, but reliable: this is why wider models can pack proportionally more nearly-orthogonal features, and why the packing gets *cleaner*, not just larger.
4. A **power law**: `L ∝ C^(−0.05)`. Operationally, the exponent is small, so returns are heavily diminishing — a 10× increase in compute multiplies loss by `10^(−0.05) ≈ 0.89`, an ~11% reduction. Every 10× buys another ~11%, forever, with no cliff and no plateau. That's the shape that makes scaling both reliably worthwhile and expensive.
5. **What's being held fixed, and what the actual constants are.** `O(S²)` is the growth in `S` with the model fixed; per-layer FLOPs also carry a `D` factor for attention versus roughly `D · d_ff` for the FFN. At the sequence lengths of most models the FFN term is larger in absolute terms despite being `O(S)` — attention only dominates once `S` becomes comparable to `d_ff`. An asymptotic exponent tells you where the crossover eventually is, never whether you're past it.

## Exercise

1. Tabulate `sin(x)`, `x`, and the relative error for `x ∈ {0.001, 0.01, 0.1, 0.5, 1.0, 2.0}`. Do the same for `cos(x)` against both `1` and `1 − x²/2`. Confirm the relative error of the first-order form scales like `x²`, and find the `x` at which each approximation crosses 1% error.
2. Sample 10,000 pairs of random unit vectors in `R^D` for `D ∈ {64, 256, 1024, 4096}` and plot the standard deviation of their cosine similarity against `D` on log-log axes. Verify the slope is `−0.5`, i.e. `1/√D`.
3. Generate `y = 3·x^(−0.08)` for `x` over four orders of magnitude, plot on log-log, and read the slope off the plot. Confirm you recover `−0.08`. Then add multiplicative noise and see how much data you need before the fitted exponent is stable to two digits — this is the practical difficulty of scaling-law fitting.
4. Take the variance-preservation argument from [2.3/01](../../part2_neural_network_fundamentals/2.3_init_normalization/01_init_variance_preservation.md) and identify explicitly which step is first-order. Then construct a case (large activations, or a nonlinearity far from its linear region) where the neglected term is not negligible.
