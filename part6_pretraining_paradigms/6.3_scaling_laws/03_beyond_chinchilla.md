# Beyond Chinchilla — Inference-Optimal Overtraining, and Where Scaling Went

Chinchilla answers "best loss per *training* FLOP." The moment you ship a model, that's the wrong objective — and the industry's answer since Llama has been to train **far** past compute-optimal on purpose. This file covers the overtraining logic and its verified magnitudes, then the broader 2024–25 shift: scaling laws stopped being one law about pretraining and became a *methodology* applied to data, post-training, and inference-time compute.

**Symbols:** `N` params, `D` tokens, `C ≈ 6ND`, per [file 01](01_power_laws_kaplan.md).

## The overtraining logic

Total lifetime cost of a model is `C_train + C_inference`, and inference scales with deployment: every served token costs `≈ 2N` FLOPs, forever. Once expected inference volume is large, the optimization flips:

> **Minimize loss subject to a fixed *serving* cost `N`** — then pour in training tokens until the marginal loss improvement stops paying.

A smaller model trained longer is worse per training-FLOP (you're on the flat part of its curve — [file 02](02_chinchilla.md)'s U-shape, deliberately off-minimum) but cheaper per served token at similar quality. Since `L(D)` at fixed `N` keeps falling as a power law ([file 01](01_power_laws_kaplan.md) — diminishing, never zero), overtraining always buys *something*; whether it pays is an economics question, and at modern serving volumes the answer has been a resounding yes.

The verified magnitudes — tokens per parameter, against Chinchilla's ~20:

| Model | `N` | `D` | `D/N` | `C = 6ND` |
|---|---|---|---|---|
| GPT-3 (2020) | 175B | 0.3T | **1.7** | 3.2e23 |
| Chinchilla (2022) | 70B | 1.4T | **20** | 5.9e23 |
| Llama-1 7B (2023) | 7B | 1.0T | 143 | 4.2e22 |
| Llama-2 7B (2023) | 7B | 2.0T | 286 | 8.4e22 |
| **Llama-3 8B (2024)** | 8B | 15T | **1,875** | 7.2e23 |
| Llama-3 70B (2024) | 70B | 15T | 214 | 6.3e24 |
| DeepSeek-V3 (2024) | 37B active | 14.8T | 400 (per active) | 3.3e24 |

Read the arc: 1.7 → 20 → 1,875 in four years. The wild row is Llama-3-8B — an 8B model that consumed **more training compute than GPT-3** (7.2e23 vs 3.2e23). Chinchilla-optimally, 7.2e23 FLOPs "should" build a ~35B model; Meta spent it on an 8B because the 8B is the one that ships to a billion devices. Two systematic notes: within one family, the *smallest* models are the most overtrained (the 8B at 1875:1, the 70B at 214:1) because small models are served most; and the DeepSeek row shows MoE's entry into this trade — overtraining per *active* parameter while total parameters carry capacity (Part 7.4).

Related same-shape moves: **distillation** (Part 9.2) — spend a big teacher's compute once, serve a small student — and **quantization** (Part 9.2), which cuts serving cost after the fact. Overtraining, distillation, and quantization are three answers to the same objective function.

## Scaling as methodology: where the laws went

The original question ("how big a model on how many tokens") is largely settled process. What survived is the *method* — fit a law on cheap runs, extrapolate to the expensive one — now applied at every stage:

- **Predicting the flagship.** Frontier reports fit internal scaling laws to forecast the big run's loss (and increasingly, benchmark scores) before committing — the direct descendant of Kaplan's fits, in production. This plus μP-style HP transfer ([2.4 supplementary](../../part2_neural_network_fundamentals/2.4_optimization/supplementary/02_setting_lr_and_schedule_across_scales.md)) is how you spend 1e26 FLOPs without a sweep.
- **Data-constrained laws** ([6.2/06](../6.2_data/06_the_data_wall.md)): Chinchilla's functional form with repetition-discounted effective tokens — allocation when `D_unique` is the binding constraint.
- **Data-mixture and filtering laws** ([6.2/03](../6.2_data/03_quality_filtering.md), [04](../6.2_data/04_mixtures_and_midtraining.md)): the constants `A, B, E` as functions of the corpus — quality shifts the curve, and that shift is itself measured by small-run fits.
- **Post-training and inference-time compute.** The o1/R1 result (Part 8.3) is routinely presented as a *new scaling axis*: performance vs. RL compute, and vs. tokens-of-thinking at inference, each tracing its own log-linear improvement. "Train-time compute is one axis among several" is the current frame — and the data wall ([6.2/06](../6.2_data/06_the_data_wall.md)) is a big reason the marginal FLOP migrated to axes that don't consume fresh corpus.

The honest caveat on the new axes: they're young fits over fewer orders of magnitude than Kaplan/Chinchilla had, measured on benchmarks rather than smooth loss ([file 04](04_emergence_debate.md)'s metric warnings apply in full), and the field's track record on fitted constants is [file 02](02_chinchilla.md)'s cautionary tale — twice.

## Why it matters in modern LLM work

- **Reading model cards:** tokens ÷ params locates any model on the Chinchilla-to-overtrained spectrum instantly, and tells you what the lab was optimizing (loss-per-training-FLOP vs. serving economics).
- **The trade is your trade too:** fine-tune-vs-API, model-size selection, distill-or-quantize decisions all inherit the `C_train`-vs-`C_inference` structure.
- **"Scaling is over" claims** are usually about one axis (pretraining `C` on found text). The methodology's answer is that the other axes — data quality, RL compute, inference compute — have their own live curves; whether *those* exponents hold is the current empirical frontier.

## Self-check

1. State the objective function that replaces Chinchilla's, and the constraint that changed.
2. Why is an overtrained model *provably* on the suboptimal side of its IsoFLOP U — and why is that fine?
3. Llama-3-8B vs. GPT-3: same order of training compute, wildly different `(N, D)`. What does each allocation reveal about its era's objective?
4. Within the Llama-3 family, why is the 8B overtrained ~9× harder (in `D/N`) than the 70B?
5. Connect the data wall to the "new scaling axes" in one causal chain.

### Answers

1. Minimize lifetime cost `C_train + (2N × tokens served)` for a target quality — equivalently, minimize loss at *fixed `N`* (the serving budget) rather than fixed `C_train`. The constraint moved from "training FLOPs are scarce" to "serving FLOPs dominate," which happens as soon as deployment volume is large.
2. At its training budget `C`, the model sits on the small-`N` (capacity-limited) arm of the U — the same `C` on a bigger model yields lower loss, by construction. It's fine because training compute isn't the scarce resource; the U-minimum answers a question (best one-shot loss per training FLOP) that a shipped product isn't asking.
3. GPT-3 (175B × 0.3T): Kaplan-era belief that parameters dominate, plus no expectation of mass serving — a research artifact's allocation. Llama-3-8B (8B × 15T): serving-cost minimization for open deployment at massive volume — a product's allocation. Same compute, opposite objectives; the `(N, D)` split is a fossil record of what each lab was optimizing.
4. Expected serving volume is inversely related to size — the 8B runs on phones, laptops, and cheap endpoints, so its lifetime inference multiple is far larger, justifying more training spend per parameter of serving cost. The general rule: optimal overtraining *increases* as expected deployments increase, so the small end of a family is always the most overtrained.
5. Compute budgets keep growing → Chinchilla-optimal allocation demands proportionally more fresh high-quality tokens → the found-text supply can't keep up ([6.2/06](../6.2_data/06_the_data_wall.md)) → the marginal FLOP flows to compute-hungry, data-light stages — synthetic generation, RL on verifiable tasks, inference-time reasoning — which then get formalized as their own scaling axes (Part 8.3).

## Exercise

Work the economics with real numbers. Take the Chinchilla-fitted loss surface (Besiroglu coefficients from [file 02](02_chinchilla.md)'s exercise) and a serving forecast of `T_serve = 1e14` tokens over a model's lifetime. (a) For quality target `L* = 2.05`, find the `(N, D)` pairs on the `L = L*` contour, and for each compute lifetime cost `6ND + 2N·T_serve`; locate the minimum and compare its `D/N` to 20. (b) Redo with `T_serve = 1e11` (a research prototype) and watch the optimum walk back toward Chinchilla. (c) At what `T_serve` does the optimal allocation for `L* = 2.05` hit 500 tokens/param? One sentence: what real-world quantity is `T_serve`, and who in a lab knows it?
