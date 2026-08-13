# Power Laws — the Kaplan Results and How to Read Them

Scaling laws are the quantitative claim that made the LLM era an *investment thesis*: loss falls as a smooth, predictable power law in model size, data, and compute. This file covers the original results (Kaplan et al., 2020) and — just as important — the reading skills: what a power law does and doesn't promise, and which of Kaplan's conclusions later fell.

**Symbols (scaling-law convention, used throughout 6.3):** `N` = parameter count (excluding embeddings, in Kaplan's fits), `D` = dataset size in **tokens**, `C` = training compute in FLOPs, `L` = cross-entropy loss in nats/token ([6.1/01](../6.1_pretraining_objectives/01_causal_lm.md)). Note the clash with this repo's `D` = model width — within 6.3, width is always `d_model`. See [NOTATION.md](../../NOTATION.md).

## The empirical facts

Train families of Transformers varying one resource with the others unconstrained, and the loss traces straight lines on log-log axes ([1.4/09](../../part1_math_foundations/1.4_optional_deeper_knowledge/09_approximations_and_orders_of_magnitude.md)) over many orders of magnitude:

```
L(N) ≈ (N_c / N)^0.076        params, data unlimited
L(D) ≈ (D_c / D)^0.095        data, params unlimited
L(C) ≈ (C_c / C)^0.050        compute, optimally allocated
```

Unpacking one of them — the other two are identical in structure. `L(N)` reads: *"the loss you end up with, as a function of how many parameters you used, when nothing else is scarce."*

- **`N` is the input** — the resource you're scaling (here, parameter count).
- **`N_c` is a fitted constant with the same units as `N`** — a reference scale that anchors *where* the curve sits, read off the data like a regression intercept. It has a literal meaning you'll never need: it's the model size at which the fit predicts loss = 1 nat exactly (since `N = N_c` makes the ratio 1, and `1^anything = 1`). Don't look for depth in it; the `_c` values are curve-anchoring constants, not quantities anyone interprets.
- **The ratio `N_c / N`** is why it's written this way rather than `const · N^−0.076`: a ratio of two model sizes is a pure number, so the units cancel and the formula stays sane regardless of whether you count `N` in millions or billions. As `N` grows past `N_c`, the ratio drops below 1, and…
- **…the exponent is the slope.** Raising a shrinking ratio to the power 0.076 shrinks the loss — *slowly*, because the exponent is tiny. On a log-log plot the exponent is literally the slope of the straight line ([1.4/09](../../part1_math_foundations/1.4_optional_deeper_knowledge/09_approximations_and_orders_of_magnitude.md)): make the model 10× bigger and loss falls by a fixed *fraction* set by the exponent alone.

So the three lines say, in plain language: with everything else unlimited, **loss falls slowly and steadily as you grow the model** (exponent 0.076); **slightly faster as you grow the data** (0.095); and when you scale *total compute* and split it well between the two, **the combined return is the smallest exponent of all** (0.050) — because compute has to buy both halves at once.

The exponents are tiny, so convert them to something you can feel — each row is verified arithmetic:

| ×10 more… | Loss multiplier | Reduction |
|---|---|---|
| `N` | `10^−0.076` = 0.840 | **16.1%** |
| `D` | `10^−0.095` = 0.803 | **19.6%** |
| `C` | `10^−0.050` = 0.891 | **10.9%** |

Three reading rules, all reused constantly:

1. **Diminishing but never-vanishing returns.** Every 10× buys the *same fraction* forever — no plateau, no cliff. That shape is what justifies "just scale": returns diminish, but predictably, so capability is purchasable ([4.1/03](../../part4_sequence_modeling_lineage/4.1_rnns_and_lstms/03_why_they_lost.md) — the property RNNs lacked).
2. **Ask what's held fixed.** `L(N)` assumes unlimited data; `L(C)` assumes you split compute optimally. Quoting an exponent without its regime is how scaling laws get misused ([1.4/09](../../part1_math_foundations/1.4_optional_deeper_knowledge/09_approximations_and_orders_of_magnitude.md)'s "what's being held fixed" rule).
3. **The floor is subtracted.** Careful fits are `L = E + (power law)` — the irreducible entropy of text `E` ([6.1/01](../6.1_pretraining_objectives/01_causal_lm.md)) sits under everything, and the power law describes only the reducible gap. Plots that bend *toward flat* on log-log axes are often just un-subtracted `E` — the line levels off as `L` approaches the floor, mimicking saturation.

## The compute identity: `C ≈ 6ND`

The accounting that links the three variables. Per token, per parameter: the forward pass costs ~2 FLOPs (one multiply + one add in some matmul), and the backward pass costs ~2× the forward (gradients w.r.t. both activations and weights) — total **6 FLOPs per parameter per token**, so a full run costs

```
C ≈ 6 · N · D
```

Good to ~10–20% for standard Transformers (attention's `S²` term adds a small correction). This identity is why "compute budget" and "choice of `(N, D)`" are the same decision — fix `C` and you're choosing a point on the hyperbola `ND = C/6` — which sets up the *allocation* question that the rest of 6.3 fights over. It's also the field's everyday estimator: Llama-3-70B at 15T tokens is `6 × 70e9 × 15e12 ≈ 6.3e24` FLOPs, and you can now price any model card in your head.

## Kaplan's allocation answer — and its famous casualty

Given `C = 6ND`, how should `N` and `D` grow as `C` grows? Kaplan's fits said: **overwhelmingly grow the model.**

```
N_opt ∝ C^0.73        D_opt ∝ C^0.27
```

Models should grow ~3× faster than data. This conclusion built **GPT-3**: 175B parameters trained on 300B tokens — **1.7 tokens per parameter** — the era of enormous, data-starved models. It was also *wrong*, for a subtle experimental reason ([file 02](02_chinchilla.md)): Kaplan's runs used a fixed LR-schedule length rather than tuning the schedule to each token budget, which systematically understated what small models could achieve with more data. The correction — equal growth, ~20 tokens/param — is Chinchilla, next file.

Two Kaplan findings that *survived*:

- **Shape barely matters.** At fixed `N`, varying depth-vs-width (`d_model/L` aspect ratio) moves loss only slightly across a broad basin — architecture's precise proportions are a second-order effect next to scale. This is the scaling-law footing for [3.2/02](../../part3_residual_connections_deep_networks/3.2_normalization_and_depth/02_scaling_the_residual_stream.md)'s ~100:1 aspect-ratio plateau.
- **Smooth loss, sample-efficient big models.** Larger models reach any given loss in fewer tokens — size buys sample efficiency, not just capacity.

## What a power law is *not*

The claims the plots don't make, kept explicit because each gets violated in the wild:

- **Not a law of nature.** It's a fitted regularity for one architecture family × one data distribution × one metric. Change the data ([6.2/03](../6.2_data/03_quality_filtering.md) — filtering shifts the constants) and the *same* plot has different numbers; the constants are where data quality lives.
- **Not about capabilities.** It predicts next-token loss. The mapping from loss to task performance can be nonlinear enough to look discontinuous — the entire emergence debate ([file 04](04_emergence_debate.md)).
- **Not extrapolation-proof.** A straight line over 6 orders is strong evidence, not proof, about the 7th — data constraints ([6.2/06](../6.2_data/06_the_data_wall.md)) and distribution shifts genuinely bend the observed curves.

## Why it matters in modern LLM work

- `C ≈ 6ND` and the 10×-table are daily mental arithmetic — pricing runs, sanity-checking model cards, reading papers' compute claims.
- The Kaplan→Chinchilla correction is the field's best cautionary tale about *experimental design in fitting laws* — the law was fine; the protocol biased the constants. [File 02](02_chinchilla.md) has a second, subtler instance of the same lesson.
- Scaling-law *methodology* (fit small, extrapolate big) is now everywhere: μP-based HP transfer ([2.4 supplementary](../../part2_neural_network_fundamentals/2.4_optimization/supplementary/02_setting_lr_and_schedule_across_scales.md)), data-mixture laws ([6.2/04](../6.2_data/04_mixtures_and_midtraining.md)), inference-compute laws ([file 03](03_beyond_chinchilla.md)).

## Self-check

1. Why does a power law with exponent −0.05 still justify spending 10× more compute, and what *shape* of curve would not?
2. Derive the 6 in `C ≈ 6ND` from the forward/backward split.
3. GPT-3's 1.7 tokens/param was downstream of which specific Kaplan conclusion — and what was the experimental flaw beneath it?
4. A paper shows loss vs. compute bending downward (super-power-law) at the large end. Give two mundane explanations to rule out before "acceleration."
5. Which two Kaplan-era findings survived Chinchilla, and where does each show up elsewhere in this repo?

### Answers

1. Because the return per 10× is *constant* (10.9% of remaining reducible loss, forever) and therefore forecastable — the spend is justified whenever a ~11% loss reduction is worth 10× the cost, which the capability market kept affirming. The disqualifying shape is an *asymptote within reach*: returns per 10× shrinking toward zero (loss flattening above `E` on log-log axes), which makes further scale strictly uneconomic.
2. Forward: each parameter participates in one multiply-accumulate per token ≈ 2 FLOPs. Backward: computing `∂L/∂activations` and `∂L/∂W` each cost another matmul of the same size ≈ 4 FLOPs. Total 6 FLOPs/param/token; multiply by `N` params and `D` tokens.
3. `N_opt ∝ C^0.73` — grow the model much faster than the data — which at GPT-3's budget prescribes a huge `N` and small `D`. The flaw: a fixed cosine-schedule length reused across token budgets, which under-trains the small-`N`/large-`D` configurations (their LR never anneals properly for their actual horizon), biasing the fit toward "params matter more" ([2.4/02](../../part2_neural_network_fundamentals/2.4_optimization/02_lr_schedules.md) — schedule length must match the run).
4. (a) **Over-subtracted floor:** if the fitted `E` is too *large*, the plotted residual `L − E_hat` heads toward zero faster than any power law and `log(L − E_hat)` plunges — manufactured acceleration (and the curve breaks entirely once the residual would go negative). (b) **Regime change in the data or eval** — e.g., the large runs used a better mixture, or the eval set overlaps training more at scale (contamination, [6.2/02](../6.2_data/02_deduplication.md)). Both mimic "scaling is accelerating" without any new physics. Note the direction: the *un*-subtracted floor produces the **opposite** artifact — plotting raw `L` flattens the line as `L → E` (local slope `−a·(L−E)/L → 0`), a fake *taper* that reads as "scaling is saturating." Floor mishandling can forge either headline; which one depends on which side of the true `E` your subtraction landed.
5. **Shape insensitivity** (broad `d_model/L` optimum) → the aspect-ratio discussion in [3.2/02](../../part3_residual_connections_deep_networks/3.2_normalization_and_depth/02_scaling_the_residual_stream.md). **Big models are sample-efficient** → the premise behind overtraining small models being *sub*optimal in training compute yet chosen anyway for inference reasons ([file 03](03_beyond_chinchilla.md) makes that trade explicit).

## Exercise

Reproduce a toy scaling law end-to-end. Generate synthetic "loss" data `L = 1.7 + 3.2·C^(−0.05)` over `C ∈ [1e18, 1e24]`, add 2% multiplicative noise. (a) Fit a naive power law to `L` directly and to `L − 1.7`; plot both on log-log and identify the bend from the un-subtracted floor. (b) Fit the exponent from only the cheapest 3 orders of magnitude and report the extrapolation error at `1e24` — this is the everyday risk of scaling-law forecasting. (c) Using `C = 6ND`, compute the FLOPs for: GPT-3 (175B × 300B), Llama-3-8B (8B × 15T), and confirm the counterintuitive one — the 8B model cost **more** than GPT-3 (7.2e23 vs 3.15e23). One sentence on what that comparison says about the post-Chinchilla era ([file 03](03_beyond_chinchilla.md)).
