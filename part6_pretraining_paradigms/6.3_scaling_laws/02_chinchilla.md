# Chinchilla — Compute-Optimal Allocation, and the Fine Print

Hoffmann et al. (2022) re-ran the allocation question with a fixed protocol flaw removed and flipped the field's answer: not "grow the model," but **grow model and data together** — about **20 tokens per parameter** at compute-optimal. This file covers the result, the demonstration, and two pieces of fine print that most summaries skip — one of which is a genuine inconsistency in the paper's own numbers, worth knowing because the corrected version changes how you read the "20:1 rule."

**Symbols:** `N` params, `D` tokens, `C ≈ 6ND` FLOPs, per [file 01](01_power_laws_kaplan.md).

## The correction

Kaplan's bias ([file 01](01_power_laws_kaplan.md)): a fixed LR-schedule length across token budgets under-trained the small-model/big-data configurations. Chinchilla tuned the cosine schedule to each run's actual horizon and measured allocation three ways:

1. **Fixed model sizes, varying data** — read the compute-optimal frontier off the envelope of training curves.
2. **IsoFLOP profiles** — for each fixed budget `C`, train many `(N, D)` pairs on the hyperbola `ND = C/6`; the loss-vs-`N` curve is U-shaped, and its minimum is the optimal allocation at that budget. (The cleanest method — worth internalizing as *the* picture of "allocation.")
3. **Parametric fit** — fit a single loss surface and optimize it analytically:

```
L(N, D) = E + A/N^α + B/D^β
```

with `E` the irreducible entropy ([6.1/01](../6.1_pretraining_objectives/01_causal_lm.md)), `A/N^α` the finite-capacity penalty, `B/D^β` the finite-data penalty. Approaches 1–2 agreed: **`N_opt ∝ C^0.5`, `D_opt ∝ C^0.5`** — scale them *together* — which at their budgets works out to `D/N ≈ 20` tokens per parameter.

## The demonstration

Same compute, reallocated: **Chinchilla (70B params × 1.4T tokens, `C ≈ 5.9e23`)** versus **Gopher (280B × 300B, `C ≈ 5.0e23`)**. The 4×-smaller, 4.7×-longer-trained model won essentially every eval — and, the underrated half, it's 4× cheaper *at inference* forever after. GPT-3 (1.7 tokens/param) was, by this account, dramatically undertrained: re-deriving from the fitted law: its `3.15e23` FLOPs, optimally reallocated (~25B params), would have reached loss ≈ 1.95 instead of ≈ 2.00 — the *entire* 175B design was inside the suboptimal region.

## Fine print 1 — why "together," structurally

The exponents `α` and `β` control everything. Optimizing `L(N, D)` under `6ND = C` gives (worth deriving once — the exercise does):

```
N_opt ∝ C^(β/(α+β))      D_opt ∝ C^(α/(α+β))      D/N ∝ C^((α−β)/(α+β))
```

Unpacking, piece by piece:

- **Where `α` and `β` live.** They're the two exponents of the fitted surface `L = E + A/N^α + B/D^β` above: `α` measures how fast the *capacity penalty* dies as the model grows; `β`, how fast the *data penalty* dies as tokens are added. A big exponent means that resource is **efficient** — a little more of it kills a lot of its penalty; a small exponent means it's **stubborn**.
- **`∝` ("proportional to") hides the constants deliberately.** The full solution is `N_opt = k · C^(β/(α+β))` with `k` assembled from `A, B, α, β` and the 6 — dropped because this section is about how allocation *scales* with budget, and constants shift log-log lines without bending them ([1.4/09](../../part1_math_foundations/1.4_optional_deeper_knowledge/09_approximations_and_orders_of_magnitude.md)).
- **Read the two exponents as budget shares.** They sum to 1 — `β/(α+β) + α/(α+β) = 1` — as they must: since `N·D ∝ C`, growth in `N` and growth in `D` split the growth in `C` between them. Each resource's *share* of every new order of magnitude of compute is a fixed fraction set by the exponents. At `α = β` the split is 50/50 — `C^0.5` each, which is exactly Chinchilla's approaches 1–2.
- **Why the swap — `N`'s share is governed by `β`, `D`'s by `α`.** The optimizer's real move (verified on the published fit): it grows `N` and `D` so that **the two penalty terms fall in lockstep** — from `C = 1e21` to `1e25`, both terms decay together as `C^−0.153` while their *ratio* stays pinned at `β/α ≈ 0.82`. Neither failure mode is ever allowed to dominate. Holding that balance forces the swap: the resource with the **stubborn** penalty (smaller own-exponent) needs the **larger** share of growth, because pouring more of it in is the only way its slow-decaying term keeps pace with the other. Stubborn capacity (small `α`) ⇒ grow `N` faster; stubborn data (small `β`) ⇒ grow `D` faster. Each exponent lands in the *other* resource's formula.
- **The third expression is just the first two subtracted.** `D/N ∝ C^((α−β)/(α+β))` — the *difference* of the shares. In plain language: the tokens-per-parameter ratio drifts with budget at a rate set by *how unequal the two penalties' decay rates are*, and in the direction of whichever resource is more stubborn.

So the token-per-parameter ratio is **scale-free if and only if `α ≈ β`** — exponent `(α−β)/(α+β) = 0`, so `D/N ∝ C^0 =` constant, one ratio for every budget. The model-size penalty and the data penalty must decay at the same rate for "20:1" to be a *constant* rather than a budget-dependent quantity. Chinchilla's approaches 1–2 found `α ≈ β ≈ 0.5`. That's the real content of the result; "20 tokens per parameter" is the memorable shadow of "`α = β` at these scales, on this data."

## Fine print 2 — the paper's own fit disagrees with its headline

Verified directly, and a good lesson in reading papers with a calculator. The published Approach-3 coefficients are `E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28`. Optimize *those* under `C = 6ND` and the optimal ratio is **not 20, and not constant**:

| `C` (FLOPs) | `N_opt` | `D_opt` | `D/N` |
|---|---|---|---|
| 1e21 | 1.8B | 0.09T | **50** |
| 1e23 | 14.6B | 1.14T | 78 |
| 6e23 (Chinchilla's own budget) | 32.7B | 3.05T | **93** |
| 1e25 | 117.5B | 14.2T | 121 |

`α ≠ β` (0.34 vs 0.28), so the ratio *grows* like `C^0.097` — the published parametric fit contradicts the paper's approaches 1–2 and its 70B/1.4T flagship run. The resolution came from **Besiroglu et al. (2024)**, who re-extracted the data and found Approach 3's fit was flawed; their refit gives `α ≈ 0.35, β ≈ 0.37` — nearly equal again — under which the optimal ratio comes out `≈ 15–19` and roughly flat across budgets, reconciling all three approaches. Two morals: the *structural* condition (`α ≈ β`) is the robust finding, the specific coefficients were not; and fitted-law papers can disagree with themselves — check the algebraic consequences of published constants before building on them.

## What "optimal" means — and doesn't

The sharpest misreading of Chinchilla is treating 20:1 as a training *target*. It is the answer to exactly one question: *"minimize final loss subject to a fixed training-compute budget, training one model, once, with no regard for what it costs to run."* Every clause is violated in practice:

- **Inference isn't free** — the entire subject of [file 03](03_beyond_chinchilla.md), and why every production model since Llama is deliberately "over-trained" far past 20:1.
- **Data isn't unlimited** — at frontier budgets, 20:1 can demand more high-quality tokens than exist ([6.2/06](../6.2_data/06_the_data_wall.md)); Muennighoff's data-constrained laws are the Chinchilla machinery re-fit with a repetition-discounted effective `D`.
- **The constants are data-dependent** — better filtering shifts `A, B, E` ([6.2/03](../6.2_data/03_quality_filtering.md)), so the optimal frontier itself moves with the corpus.

Chinchilla's durable contribution is the *frame*: allocation is a measurable optimization with a clean methodology (IsoFLOPs), not a taste question. The specific ratio is an artifact of its assumptions.

## Why it matters in modern LLM work

- **"Chinchilla-optimal" is live vocabulary** — usually invoked to say a model is far from it, which you can now compute: tokens ÷ params, compare to ~20.
- **The IsoFLOP method** is the reusable instrument — it reappears for MoE sizing (Part 7.4), vocabulary sizing, and inference-compute allocation.
- **The Besiroglu episode** is the field's standing reminder that scaling "laws" are fits with error bars, produced by fallible protocols — twice now (Kaplan's schedules, Chinchilla's Approach 3) the correction mattered at billion-dollar scale.

## Self-check

1. What single experimental-protocol change separates Chinchilla from Kaplan, and why did it flip the conclusion?
2. Explain the IsoFLOP method and what the U-shape's two arms represent.
3. Under what algebraic condition is "tokens per parameter" scale-free, and what happens to the ratio when `α > β`?
4. Gopher and Chinchilla cost similar training compute. Name *both* ways Chinchilla wins economically.
5. Give three distinct reasons a 2025 frontier lab deliberately trains far from 20:1.

### Answers

1. Tuning the LR-decay horizon to each run's token budget instead of reusing one fixed schedule. Under the fixed schedule, small-`N` large-`D` runs never annealed properly for their horizon and looked artificially bad, biasing the fit toward "spend on parameters"; with matched schedules their true (much better) performance surfaced, moving the optimum toward data.
2. Fix `C`; train many `(N, D)` pairs with `6ND = C`; plot final loss against `N`. Left arm (small `N`): capacity-limited — the model can't use its many tokens (`A/N^α` dominates). Right arm (large `N`): data-limited — too few tokens to train the many parameters (`B/D^β` dominates). The minimum is the budget's optimal allocation, and tracking it across several `C` values traces `N_opt(C)` directly, no parametric form needed.
3. Scale-free iff `α = β`, since `D/N ∝ C^((α−β)/(α+β))`. With `α > β` (model-size penalty decays faster), the optimizer shifts budget toward data as `C` grows — the ratio *rises* with scale, exactly what the flawed published fit (0.34 vs 0.28) implies, and why its `D/N` runs 50→121 instead of sitting at 20.
4. (a) At equal training compute it reaches lower loss / better evals — the headline; (b) it's 4× smaller, so every future forward pass costs ~4× less — the compounding win, and the observation [file 03](03_beyond_chinchilla.md) pushes to its conclusion (if inference dominates lifetime cost, go *smaller than optimal* and train even longer).
5. (a) **Inference economics** — lifetime serving cost dwarfs training, favoring small-and-overtrained ([file 03](03_beyond_chinchilla.md)); (b) **data supply** — 20:1 at a 1e26-FLOP budget demands more quality tokens than exist, forcing repetition-adjusted allocation ([6.2/06](../6.2_data/06_the_data_wall.md)); (c) **the constants moved** — with heavily filtered/synthetic data the fitted `A, B, E` differ from Chinchilla's MassiveText, so its numeric optimum simply isn't theirs (also acceptable: deployment memory caps, distillation plans).

## Exercise

Do the derivation and then the audit. (a) Minimize `L = E + A/N^α + B/D^β` subject to `6ND = C` (Lagrange or substitution) and derive `N_opt ∝ C^(β/(α+β))` and the `D/N ∝ C^((α−β)/(α+β))` corollary. (b) Implement the published coefficients and reproduce the table above — confirm `D/N ≈ 93` at Chinchilla's own 6e23 budget, i.e., the fit recommends against the paper's flagship run. (c) Swap in the Besiroglu refit (`E=1.82, A=482, B=2085, α=0.35, β=0.37`) and confirm the ratio flattens to ~15–19. (d) One sentence: which single scalar in each fit did all the work of changing the conclusion?
