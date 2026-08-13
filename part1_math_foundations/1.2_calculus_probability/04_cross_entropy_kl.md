# Cross-Entropy and KL Divergence — Why CE Is The Loss

## Entropy

For a discrete distribution `p`:
```
H(p) = -Σ p_i log p_i
```

Measures uncertainty. Max when uniform, 0 when one-hot. Units:
- log base 2 → bits
- log base e → nats
- log base 256 → bytes

In ML we almost always use nats (natural log) for math and report bits or perplexity in practice.

For the natural-language picture — what `−log(p_i)` means as "surprise," why you weight by `p_i`, the two extremes, concrete numbers, and the Shannon-coding operational meaning — see `supplementary/04_entropy_intuition.md`.

## Cross-entropy

For a *true* distribution `p` and a *predicted* distribution `q`:
```
H(p, q) = -Σ p_i log q_i
```

"The expected surprise (measured under `p`) when you assume `q`." It's always ≥ `H(p)`, with equality iff `p = q`.

In plain words: **surprise is computed from your beliefs (`−log q_i`), but averaged over reality's frequencies (`p_i`)** — so the closer your assumed `q` matches the true `p`, the less surprised you'll be on average. The minimum, `H(p)`, is the irreducible surprise you pay even with a perfect model; the excess `H(p, q) − H(p) = KL(p ‖ q)` is the tax for guessing wrong. Training a model to minimize cross-entropy is literally training it to stop being surprised by real data.

In language modeling: `p` is the one-hot true next token, `q` is the model's predicted distribution. Then:
```
H(p, q) = -log q_{true_token}
```

The loss for a single token is just `-log` of the probability the model assigned to the correct answer. Total loss is the sum (or mean) over all token positions.

## Negative log-likelihood (NLL) — the same thing, named from the other side

**What it is.** For a single labeled example, NLL is `−log q_c` — the negative log of the probability your model assigned to the *observed* outcome `c`. Over a dataset of independent examples it's the sum:
```
NLL = −Σ_n log q(observed_n)
```

**Why it's identical to the CE you just saw.** Plug a one-hot target into cross-entropy `H(p, q) = −Σ_i p_i log q_i` and only the true class survives, leaving `−log q_c`. So **per-example CE against a hard label *is* the NLL of that label.** "Cross-entropy loss," "log loss," and "negative log-likelihood" are three names for the same number in the categorical/LM setting. The CE name emphasizes "distance between two distributions"; the NLL name emphasizes "how probable was the data."

**What it represents.** Start from the *likelihood* — the probability the model assigns to the data you actually observed: `L = Π_n q(observed_n)`. You want this high (a good model finds reality probable). Two moves convert it into a loss:
- **Take the log.** Products of thousands of tiny probabilities underflow to zero; `log` turns the product into a sum (`log L = Σ log q`), which is numerically stable and additive across examples.
- **Negate.** Maximizing likelihood becomes *minimizing* `−log L`, so it behaves like a loss (lower = better) that gradient descent can drive down.

So NLL is "**the total surprise of the data under the model**," in nats. Each term `−log q` is the surprise of one observation; low when the model expected it (`q→1`, surprise `→0`), exploding when the model ruled it out (`q→0`, surprise `→∞`).

**Why it's used.**
- **It's maximum likelihood.** Minimizing NLL = maximizing `log p_model(data)` = MLE, the principled statistical objective for fitting a probability model. This is the same identity listed under "Why CE is the right loss" below.
- **It scores the whole distribution, not just the argmax.** Unlike accuracy, NLL cares *how confident* you were. Being right at 0.9 beats being right at 0.51; being wrong at 0.99 is punished savagely. That gives a smooth, informative gradient everywhere — exactly what training needs.
- **It composes cleanly with softmax.** As the gradient section shows, `log_softmax` + NLL gives `∂L/∂logits = predicted − target` with no chain-rule mess, and is numerically stable. (`F.cross_entropy` in PyTorch *is* `log_softmax` + `nll_loss` fused.)
- **It's interpretable.** Mean NLL per token in nats → exponentiate for perplexity, convert with `log₂(e)` for bits/byte. The loss you train on is the same quantity you report at eval (see the perplexity and bits-per-byte files).

The harsh `q→0 ⇒ loss→∞` behavior is a feature: it forbids the model from ever assigning *zero* probability to something that actually happens. Confident-and-wrong is the most expensive mistake under NLL.

## What CE loss actually does (loss value vs. gradient)

For LMs and classification, the target distribution `p` is one-hot — there's a single correct class `c` with `p_c = 1` and every other entry zero. Two distinct questions are worth separating: *what does the loss value depend on?* and *what does the backward pass actually correct?* The answers are different, and the difference is the whole story.

**The loss value: only the true class contributes.** Plug a one-hot target into `H(p, q) = −Σ p_i log q_i` and every term with `p_i = 0` multiplies `log q_i` by zero and vanishes. You're left with a single survivor: `−log(q_c)`, the negative log of the probability the model assigned to the correct class. Maxing out `q_c` drives the loss to zero; sending `q_c → 0` sends the loss to infinity. So the *number* you'd print as "the loss" is solely a function of how much probability went to the right answer — wrong-class probabilities don't appear in the formula at all.

**But wrong classes still matter, indirectly.** `q_c` is the output of a softmax, which forces all probabilities to sum to 1. Mass placed on a wrong class is mass *stolen* from `q_c`. So putting weight on wrong tokens shrinks `q_c`, which inflates `−log(q_c)`, which raises the loss. The wrong classes influence the loss value through normalization, one step removed.

**The gradient: every class gets corrected.** The gradient of CE+softmax w.r.t. logits is `∂L/∂x = s − y` — predicted probabilities minus target probabilities — a vector with one entry per class. Read entry-by-entry:

- **Correct class** (target = 1): gradient is `s_c − 1`. Negative, so backprop pushes the logit *up*. Magnitude `1 − s_c` — exactly how much probability the right class is still missing from full confidence.
- **Each wrong class** (target = 0): gradient is `s_j − 0 = s_j`. Positive, so backprop pushes the logit *down*. Magnitude `s_j` — exactly how much probability the model wrongly put there. The bigger the offender, the harder the push-down. A wrong class at 30% gets a hefty 0.3 correction; a wrong class at 0.1% gets a tiny 0.001 nudge.

**The correction is balanced.** Sum of upward push on the correct class: `1 − s_c`. Sum of downward pushes on wrong classes: `Σ_{j ≠ c} s_j = 1 − s_c` (since all probabilities sum to 1). Equal magnitudes. Probability mass flows from wrong classes to the right one in exact proportion to how much each was stealing.

**Concrete example.** Four classes, true class is class 1, softmax outputs are `[0.5, 0.3, 0.15, 0.05]`.
- Loss value: `−log(0.5) ≈ 0.693`. Only `s_1` enters this.
- Gradient on logits: `[0.5 − 1, 0.3, 0.15, 0.05] = [−0.5, 0.3, 0.15, 0.05]`.
- Backward effect: push logit 1 up by 0.5; push logit 2 down by 0.3 (biggest offender → biggest correction); logit 3 down by 0.15; logit 4 down by 0.05. Upward push = 0.5, downward pushes = 0.5. Balanced.

**TL;DR.** Loss value reads only the correct class's probability. Gradient corrects *every* class — pushes the right one up by how-much-it's-missing, pushes each wrong one down by how-much-mass-it-currently-holds. The two halves are equal in total magnitude, which is why backprop on CE+softmax feels like a clean redistribution rather than a separate "reward right / punish wrong" computation.

## KL divergence

```
KL(p ‖ q) = Σ p_i log(p_i / q_i) = H(p, q) - H(p)
```

- Always ≥ 0, zero iff `p = q`.
- **Not symmetric**: `KL(p ‖ q) ≠ KL(q ‖ p)`.
- Cross-entropy minimization is equivalent to KL minimization when `p` is fixed (which it is in supervised learning — the data distribution doesn't depend on model params).

**Forward vs. reverse KL** is a thing you should understand:
- `KL(p ‖ q)` (forward, "mean-seeking"): penalizes `q` being small where `p` is large. Makes `q` cover all modes of `p`. This is what standard maximum-likelihood training does.
- `KL(q ‖ p)` (reverse, "mode-seeking"): penalizes `q` being large where `p` is small. Makes `q` concentrate on high-probability regions of `p`. This is what variational methods and RL-from-preferences use.

DPO, PPO with KL penalty, and RLHF all involve KL terms — paying attention to *which direction* is the whole game.

**Where each direction shows up in practice:**

- **LM pretraining / MLE** → *forward KL*. Minimizing CE on data is exactly minimizing `KL(p_data ‖ p_model)`. Mode-covering: the model must put non-zero mass on everything in the training corpus → broad capabilities, "averaged" outputs.
- **Diffusion training** → *forward KL* (between forward and reverse processes). Mode-covering is why diffusion gives diverse generations and resists the mode collapse that plagued GANs.
- **RLHF / PPO with KL penalty** → *reverse KL*: `KL(π_policy ‖ π_ref)`. Mode-seeking: the policy concentrates on one high-reward response style → committed personality, repetitive outputs ("RLHF mode collapse" is reverse-KL working as designed).
- **DPO** → same reverse-KL structure as RLHF, baked in analytically — the closed-form derivation *assumes* a reverse-KL constraint.
- **VAEs / variational inference** → *reverse KL*: `KL(q ‖ p_posterior)`. Two reasons: you can sample from `q` but not from the intractable posterior, and reverse KL forces `q` to live inside `p`'s support (sharp, committed approximation).

The recipe SFT → RLHF works precisely because the two phases use opposite directions: pretraining/SFT uses forward KL for breadth, RLHF uses reverse KL for style. Swap either, and you break the pipeline.

## Why CE is the right loss for LMs

Three equivalent framings, all true:

1. **Maximum likelihood**: minimizing CE = maximizing `log p_model(data)`.
2. **KL minimization**: minimizing CE = minimizing `KL(data ‖ model)` (plus a constant).
3. **Compression**: minimizing CE = finding the model that compresses the data best (see next file on perplexity / bits per byte).

All three say the same thing. Different communities prefer different framings.

## MSE vs. CE — a common beginner confusion

For classification / LM, **always use CE on logits**. Never:
- MSE on probabilities (wrong gradient shape, near-zero gradient at saturation).
- CE on probabilities (you'd do log twice).
- MSE on logits (ignores the probabilistic structure).

The combination `log_softmax` + `nll_loss` (i.e. `F.cross_entropy`) is correct and numerically stable. Use the fused op.

## The clean gradient

As noted in the softmax file: for CE loss with softmax output and one-hot target:
```
∂L/∂logits = softmax(logits) - one_hot(target)
         = predicted_probs - target_probs
```

No chain-rule mess — the softmax and CE derivatives cancel cleanly. This is what makes training LMs feel like "just predict the next token and update toward it." Deep and simple at once.

## KL penalties in RLHF/DPO

In policy optimization (PPO for RLHF) you typically add a KL term to the loss:
```
L = -E[reward] + β · KL(π_θ ‖ π_ref)
```

Keeps the updated policy `π_θ` close to the reference (SFT) model `π_ref`. Without it, PPO can catastrophically collapse into reward hacks. The `β` controls how conservative the update is.

In DPO the same KL structure appears, but integrated analytically — the KL constraint is what lets you solve the optimal policy in closed form and back out an implicit reward.

## Self-check

1. Why is `KL(p ‖ q) ≥ 0`? (Hint: Jensen's inequality on `log`.)
2. Which direction of KL is "zero-avoiding" and which is "zero-forcing"? Why does this matter for generative models?
3. Given vocab size 50K and a model achieving mean CE = 2.3 nats per token on held-out data, what's its perplexity?

### Answers

1. `KL(p ‖ q) = E_p[log(p/q)] = -E_p[log(q/p)]`. By Jensen's inequality (since `log` is concave): `E_p[log(q/p)] ≤ log(E_p[q/p]) = log(Σ p · q/p) = log(Σ q) = log 1 = 0`. Negate: `-E_p[log(q/p)] ≥ 0`, so `KL(p ‖ q) ≥ 0`. Equality iff `q/p` is constant, which forces `p = q`.
2. **Forward `KL(p ‖ q)`** = **zero-avoiding** (mode-covering). Penalizes `q` being small wherever `p` is large → `q` must spread across all of `p`'s mass. Risk: `q` smears across multiple modes, producing a "blurry average." **Reverse `KL(q ‖ p)`** = **zero-forcing** (mode-seeking). Penalizes `q` being large where `p` is small → `q` concentrates on one of `p`'s modes, ignoring others. Risk: misses valid modes. **For generative models**: standard MLE training (next-token prediction) does forward KL → covers everything → "averaged" outputs at the cost of sometimes producing low-probability text. Variational methods, RL with KL penalty, and reverse-KL distillation can do mode-seeking → sharper, more committal outputs but less diverse.
3. `PPL = e^{2.3} ≈ 9.97`. So roughly **10** — the model is "as uncertain as if choosing uniformly among 10 next tokens" (vs. the uniform baseline of 50K). A meaningful improvement over uniform (5000× reduction in uncertainty), reasonable for a small/midsize LM. Frontier LMs on clean text are typically 3–5.

## Exercise

Implement CE loss from scratch in PyTorch two ways: (a) via `softmax` → `log` → `gather`, (b) via `log_softmax` → `gather`. Feed in large logits (~1000) and compare. You should see (a) explode/NaN and (b) be fine. Then compare both against `F.cross_entropy` to confirm.
