# Emergent Abilities — and the Mirage Rebuttal

Scaling laws promise smoothness: loss falls predictably ([file 01](01_power_laws_kaplan.md)). Yet the era's most-cited capability claim was about *discontinuity* — abilities that are absent, absent, absent, then suddenly present at some scale. Both observations are about the same models, so at most one framing can be fundamental. This file states the claim, the rebuttal, and the calibrated position — which matters practically, because it decides whether you believe capabilities can be *forecast*.

## The claim (Wei et al., 2022)

**Emergent abilities:** performance on some tasks (multi-step arithmetic, word unscrambling, certain MMLU categories) is at chance for models below a scale threshold, then climbs steeply above it — "not present in smaller models but present in larger models," unpredictable in advance. The signature plot: task accuracy vs. log-compute, flat then hockey-sticking, across several model families.

Why it landed: it matches phenomenology (GPT-3.5 → GPT-4 *felt* discontinuous), and it has a plausible mechanism — some capabilities are **compositional**, requiring several sub-skills simultaneously, so task success arrives only when the *last* prerequisite matures. It also carried an unsettling implication: if capabilities appear unpredictably, safety-relevant ones might too, and small-scale experiments can't de-risk large runs.

## The rebuttal (Schaeffer et al., 2023 — "a mirage?")

The counter-observation is about **metrics, not models**. Most claimed emergence used *discontinuous* metrics — exact-match accuracy, multiple-choice — which award 0 until the model is entirely right. Take 5-digit arithmetic under exact match: per-token probability of the correct digits can improve smoothly for orders of magnitude while `P(all tokens correct) ≈ p^k` stays negligible, then rockets once per-token `p` nears 1. The *underlying* quantity was a smooth power law the whole time; the threshold lives in the scoring rule.

Their evidence cuts both ways cleanly: switch emergent tasks to continuous metrics (token-level likelihood, edit distance, partial credit) and the discontinuities largely dissolve into smooth curves; conversely, impose a discontinuous metric on a task known to scale smoothly and you *manufacture* emergence on demand. The nonlinear-metric mechanism is real, demonstrated, and explains a large fraction of the canonical examples.

## The calibrated position

Not "emergence is fake" — the correct summary is narrower and more useful:

- **Sharp thresholds are usually metric artifacts; the underlying competence is usually smooth.** Default expectation: log-likelihood of correct behavior improves as a power law, and your step function is your scoring rule ([1.4/09](../../part1_math_foundations/1.4_optional_deeper_knowledge/09_approximations_and_orders_of_magnitude.md): always ask what's being plotted).
- **Some discontinuities survive the metric critique.** The clean counterexample is mechanistic: **induction heads** form in an abrupt phase change during training, visible as a bump in the *loss itself* — smooth metric, discontinuous dynamics (Part 11.2). Grokking is a second example. Phase transitions in learned circuits are real; they're just rarer than the 2022 plots suggested — and note both examples are discontinuities in *training time*, where the claimed emergence was across *scale*.
- **"Smooth" ≠ "forecastable in practice."** Even granting smoothness of the underlying quantity, the mapping from loss to *the metric you actually care about* (pass rate, user preference) can be so steep that forecasting stays hard exactly where it matters. The mirage paper relocates the difficulty from physics to metric design; it doesn't abolish it.

## Practical consequences

This debate has an unusually direct payoff — it's really a *measurement discipline*:

- **Forecast on smooth quantities.** Labs predicting flagship capability ([file 03](03_beyond_chinchilla.md)) fit laws to continuous proxies (likelihood on task answers, per-token accuracy) and only then translate to headline metrics. Fitting a law to exact-match numbers is fitting a threshold artifact.
- **Read capability plots with the metric in hand.** Hockey-stick + exact-match ⇒ assume mirage until shown otherwise; discontinuity in *loss* ⇒ genuinely interesting.
- **Eval design** (Part 11.1): benchmarks scored by partial credit / likelihood give scaling signal *before* the pass-rate threshold; all-or-nothing scores hide approaching capabilities until they arrive — the difference between a leading and a trailing indicator, which is also why the safety-forecasting question stays open even under the mirage view.

## Why it matters in modern LLM work

- It's the standing example of **loss ≠ capability**: 6.3's smooth machinery and the jagged benchmark reality are reconciled through metric shape, closing the loop that [file 01](01_power_laws_kaplan.md) opened ("not about capabilities").
- Emergence claims persist for each new axis (RL compute, inference-time thinking — Part 8.3); the same two-question audit applies each time: *what's the metric, and is the underlying likelihood smooth?*
- The one place discontinuity is well-documented — circuit formation, induction heads — is a Part 11.2 topic, and the honest answer to "can anything truly emerge?"

## Self-check

1. Reproduce the arithmetic argument: why does exact-match on `k`-token answers manufacture a threshold from smooth per-token improvement?
2. What two symmetric demonstrations made the mirage paper convincing, rather than just plausible?
3. Give the standard counterexample to "all emergence is metric artifact," and say precisely what's discontinuous in it.
4. Your team's eval shows a capability "emerging" between the 30B and 70B checkpoints. What two plots do you make before reporting emergence?
5. Why does the mirage view *not* fully rescue capability forecasting?

### Answers

1. `P(answer) ≈ p^k` for per-token probability `p`. If `p` rises smoothly (say 0.5 → 0.9 → 0.99 across scales), `p^k` for `k = 10` goes 0.001 → 0.35 → 0.90 — negligible, negligible, then most of the mass in one step. Exact match applies a product-then-threshold nonlinearity to a smooth input; the "jump" is `p^k`'s geometry, not the model's ([1.4/09](../../part1_math_foundations/1.4_optional_deeper_knowledge/09_approximations_and_orders_of_magnitude.md)).
2. (a) Dissolution: re-scoring canonically "emergent" tasks with continuous metrics turned jumps into smooth curves; (b) construction: imposing discontinuous metrics on known-smooth tasks *created* emergence where none was claimed. Ability to both remove and induce the phenomenon by touching only the metric is strong evidence the metric was the cause.
3. Induction-head formation (Part 11.2): during training, in-context-learning ability and its associated circuit appear in a narrow window, visible as a bump/drop in the *training loss* — a metric with no threshold nonlinearity. What's discontinuous is the learning *dynamics* (a phase change in the computation the network implements), not the scoring — though note it's a discontinuity across training time, a related-but-distinct claim from emergence across model scale.
4. (a) The same eval re-scored with a continuous metric (per-token likelihood of correct answers / partial credit) across all checkpoints — if smooth, you have a threshold artifact plus a steep metric, not emergence; (b) the underlying loss (or likelihood) curve itself across checkpoints, checking for any discontinuity in the smooth quantity. Only a jump that survives both plots deserves the word.
5. Because decisions are made in the metric that jumped, not the likelihood that didn't: knowing per-token likelihood improves smoothly doesn't tell you *at which scale* `p^k` crosses usefulness for your task, unless you can fit and extrapolate the likelihood curve precisely — and near the steep region, small fit errors are large threshold-location errors. Smoothness makes forecasting *possible in principle*; it leaves it hard in practice exactly near the transitions people care about.

## Exercise

Build the mirage in 30 lines. Model per-token accuracy as `p(C) = 1 − 0.9·C^(−0.07)` (smooth power-law improvement) over `C ∈ [1e20, 1e26]`. (a) Plot `p(C)` — boring and smooth. (b) Plot exact-match `p^k` for answer lengths `k ∈ {1, 5, 20}` on the same axes: watch thresholds appear and *move right* with `k`, and note the prediction this makes — longer-answer tasks "emerge" later at identical underlying competence. (c) Plot a continuous alternative (mean per-token accuracy on the answer, i.e. `p` itself) and confirm it "de-emerges" the task. (d) Two sentences: your lab leader asks "will the 10× run get task X?" — using these plots, say what you'd fit, what you'd report, and where the honest error bars blow up.
